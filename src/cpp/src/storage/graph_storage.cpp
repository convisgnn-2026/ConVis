//
// Created by Jason Mohoney on 6/18/21.
//

#include "storage/graph_storage.h"
#include "storage/bucket_mapping_profiler.h"

#include <algorithm>
#include <random>
#include <chrono>
#include <unordered_map>

#include "data/ordering.h"
#include "reporting/logger.h"

// Global timing collector for data loading analysis
static std::unordered_map<std::string, double> timing_accumulator;
static std::unordered_map<std::string, int> timing_counter;
static std::mutex timing_mutex;

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, shared_ptr<StorageConfig> storage_config) {
    storage_ptrs_ = storage_ptrs;
    train_ = true;
    full_graph_evaluation_ = storage_config->full_graph_evaluation;

    prefetch_ = storage_config->prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    num_nodes_ = storage_config->dataset->num_nodes;
    num_edges_ = storage_config->dataset->num_edges;

    if (full_graph_evaluation_) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings)) {
                string node_embedding_filename = storage_config->model_dir + PathConstants::embeddings_file + PathConstants::file_ext;

                in_memory_embeddings_ =
                    std::make_shared<InMemory>(node_embedding_filename, storage_ptrs_.node_embeddings->dim0_size_, storage_ptrs_.node_embeddings->dim1_size_,
                                               storage_ptrs_.node_embeddings->dtype_, torch::kCPU);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {
                string node_feature_filename =
                    storage_config->dataset->dataset_dir + PathConstants::nodes_directory + PathConstants::features_file + PathConstants::file_ext;

                in_memory_features_ = std::make_shared<InMemory>(node_feature_filename, storage_ptrs_.node_features->dim0_size_,
                                                                 storage_ptrs_.node_features->dim1_size_, storage_ptrs_.node_features->dtype_, torch::kCPU);
            }
        }
    }
}

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, bool prefetch) {
    storage_ptrs_ = storage_ptrs;
    train_ = true;
    full_graph_evaluation_ = false;

    prefetch_ = prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    if (storage_ptrs_.node_embeddings != nullptr) {
        num_nodes_ = storage_ptrs_.node_embeddings->getDim0();
    } else if (storage_ptrs_.node_features != nullptr) {
        num_nodes_ = storage_ptrs_.node_features->getDim0();
    } else {
        throw MariusRuntimeException("The input graph must have node features and/or node embeddings");
    }
    num_edges_ = storage_ptrs_.edges->getDim0();
}

GraphModelStorage::~GraphModelStorage() {
    unload(false);

    delete subgraph_lock_;
    delete subgraph_cv_;
}

void GraphModelStorage::_load(shared_ptr<Storage> storage) {
    if (storage != nullptr) {
        storage->load();
    }
}

void GraphModelStorage::_unload(shared_ptr<Storage> storage, bool write) {
    if (storage != nullptr) {
        storage->unload(write);
    }
}

void GraphModelStorage::load() {
    _load(storage_ptrs_.edges);
    _load(storage_ptrs_.train_edges);
    _load(storage_ptrs_.train_edges_dst_sort);
    _load(storage_ptrs_.nodes);

    if (train_) {
        _load(storage_ptrs_.node_embeddings);
        _load(storage_ptrs_.node_optimizer_state);
        _load(storage_ptrs_.node_features);
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings) && full_graph_evaluation_) {
                _load(in_memory_embeddings_);
            } else {
                _load(storage_ptrs_.node_embeddings);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features) && full_graph_evaluation_) {
                _load(in_memory_features_);
            } else {
                _load(storage_ptrs_.node_features);
            }
        }
    }

    _load(storage_ptrs_.encoded_nodes);

    _load(storage_ptrs_.node_labels);
    _load(storage_ptrs_.relation_features);
}

void GraphModelStorage::unload(bool write) {
    _unload(storage_ptrs_.edges, false);
    _unload(storage_ptrs_.train_edges, false);
    _unload(storage_ptrs_.train_edges_dst_sort, false);
    _unload(storage_ptrs_.validation_edges, false);
    _unload(storage_ptrs_.test_edges, false);
    _unload(storage_ptrs_.nodes, false);
    _unload(storage_ptrs_.train_nodes, false);
    _unload(storage_ptrs_.valid_nodes, false);
    _unload(storage_ptrs_.test_nodes, false);
    _unload(storage_ptrs_.node_embeddings, write);
    _unload(storage_ptrs_.encoded_nodes, write);
    _unload(storage_ptrs_.node_optimizer_state, write);
    _unload(storage_ptrs_.node_features, false);
    _unload(storage_ptrs_.relation_features, false);

    _unload(in_memory_embeddings_, false);
    _unload(in_memory_features_, false);

    for (auto f_edges : storage_ptrs_.filter_edges) {
        _unload(f_edges, false);
    }

    active_edges_ = torch::Tensor();
    active_nodes_ = torch::Tensor();
}

void GraphModelStorage::setEdgesStorage(shared_ptr<Storage> edge_storage) { storage_ptrs_.edges = edge_storage; }

void GraphModelStorage::setNodesStorage(shared_ptr<Storage> node_storage) { storage_ptrs_.nodes = node_storage; }

EdgeList GraphModelStorage::getEdges(Indices indices) {
    if (active_edges_.defined()) {
        return active_edges_.index_select(0, indices);
    } else {
        return storage_ptrs_.edges->indexRead(indices);
    }
}

EdgeList GraphModelStorage::getEdgesRange(int64_t start, int64_t size) {
    if (active_edges_.defined()) {
        return active_edges_.narrow(0, start, size);
    } else {
        return storage_ptrs_.edges->range(start, size);
    }
}

void GraphModelStorage::shuffleEdges() { storage_ptrs_.edges->shuffle(); }

// Method to return random node IDs
// Random node selection functionality used for negative sampling, etc.
Indices GraphModelStorage::getRandomNodeIds(int64_t size) {
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(storage_ptrs_.edges->device_);

    Indices ret;
    if (useInMemorySubGraph()) {  // When using partition buffer storage
        if (storage_ptrs_.node_embeddings != nullptr) {
            // Extract random IDs from node embedding storage
            ret = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getRandomIds(size);
        } else {
            // Extract random IDs from node feature storage
            ret = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getRandomIds(size);
        }
    } else {
        // Random selection from entire in-memory node range
        ret = torch::randint(getNumNodesInMemory(), {size}, ind_opts);
    }

    return ret;
}

Indices GraphModelStorage::getNodeIdsRange(int64_t start, int64_t size) {
    if (active_nodes_.defined()) {
        return active_nodes_.narrow(0, start, size);
    } else {
        return storage_ptrs_.nodes->range(start, size).flatten(0, 1);
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddings(Indices indices) {
    if (!train_ && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings) && full_graph_evaluation_) {
        if (in_memory_embeddings_ != nullptr) {
            return in_memory_embeddings_->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return storage_ptrs_.node_embeddings->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddingsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getEncodedNodes(Indices indices) {
    if (storage_ptrs_.encoded_nodes != nullptr) {
        return storage_ptrs_.encoded_nodes->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getEncodedNodesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.encoded_nodes != nullptr) {
        return storage_ptrs_.encoded_nodes->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeFeatures(Indices indices) {
    if (!train_ && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features) && full_graph_evaluation_) {
        if (in_memory_features_ != nullptr) {
            return in_memory_features_->indexRead(indices);

        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_features != nullptr) {
            return storage_ptrs_.node_features->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor GraphModelStorage::getNodeFeaturesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeLabels(Indices indices) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeLabelsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddings(Indices indices, torch::Tensor embeddings) { storage_ptrs_.node_embeddings->indexPut(indices, embeddings); }

void GraphModelStorage::updateAddNodeEmbeddings(Indices indices, torch::Tensor values) { storage_ptrs_.node_embeddings->indexAdd(indices, values); }

void GraphModelStorage::updatePutEncodedNodes(Indices indices, torch::Tensor values) { storage_ptrs_.encoded_nodes->indexPut(indices, values); }

void GraphModelStorage::updatePutEncodedNodesRange(int64_t start, int64_t size, torch::Tensor values) {
    storage_ptrs_.encoded_nodes->rangePut(start, size, values);
}

OptimizerState GraphModelStorage::getNodeEmbeddingState(Indices indices) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

OptimizerState GraphModelStorage::getNodeEmbeddingStateRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddingState(Indices indices, OptimizerState state) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexPut(indices, state);
    }
}

void GraphModelStorage::updateAddNodeEmbeddingState(Indices indices, torch::Tensor values) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexAdd(indices, values);
    }
}

bool GraphModelStorage::embeddingsOffDevice() {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->device_ != torch::kCUDA;
    } else if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->device_ != torch::kCUDA;
    } else {
        return false;
    }
}

void GraphModelStorage::initializeInMemorySubGraph(torch::Tensor buffer_state, int num_hash_maps) {
    if (useInMemorySubGraph()) {
        current_subgraph_state_ = std::make_shared<InMemorySubgraphState>();

        buffer_state = buffer_state.to(torch::kInt64);

        int buffer_size = buffer_state.size(0);
        int num_edge_buckets_in_mem = buffer_size * buffer_size;
        int num_partitions = getNumPartitions();

        torch::Tensor new_in_mem_partition_ids = buffer_state;
        auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

        torch::Tensor in_mem_edge_bucket_ids = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor in_mem_edge_bucket_sizes = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor global_edge_bucket_starts = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);

        auto in_mem_edge_bucket_ids_accessor = in_mem_edge_bucket_ids.accessor<int64_t, 1>();
        auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
        auto global_edge_bucket_starts_accessor = global_edge_bucket_starts.accessor<int64_t, 1>();

        // TODO we don't need to do this every time
        auto bucket_prep_start = std::chrono::high_resolution_clock::now();
        std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
        torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int)edge_bucket_sizes_.size()}, torch::kInt64);
        torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
        torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();
        auto bucket_prep_end = std::chrono::high_resolution_clock::now();
        double bucket_prep_time = std::chrono::duration<double, std::milli>(bucket_prep_end - bucket_prep_start).count();

        auto bucket_mapping_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < buffer_size; i++) {
            for (int j = 0; j < buffer_size; j++) {
                int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];
                int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
                int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

                int idx = i * buffer_size + j;
                in_mem_edge_bucket_ids_accessor[idx] = edge_bucket_id;
                in_mem_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
                global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
            }
        }
        auto bucket_mapping_end = std::chrono::high_resolution_clock::now();
        double edge_bucket_metadata_time = std::chrono::duration<double, std::milli>(bucket_mapping_end - bucket_mapping_start).count();

        torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
        int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
        in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

        auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

        current_subgraph_state_->all_in_memory_edges_ = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

        auto edge_loading_start = std::chrono::high_resolution_clock::now();
        int64_t total_edges_loaded = 0;
#pragma omp parallel for reduction(+:total_edges_loaded)
        for (int i = 0; i < num_edge_buckets_in_mem; i++) {
            int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = global_edge_bucket_starts_accessor[i];
            int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];
            total_edges_loaded += edge_bucket_size;

            current_subgraph_state_->all_in_memory_edges_.narrow(0, local_offset, edge_bucket_size) =
                storage_ptrs_.edges->range(edge_bucket_start, edge_bucket_size);
        }
        auto edge_loading_end = std::chrono::high_resolution_clock::now();
        double edge_loading_time = std::chrono::duration<double, std::milli>(edge_loading_end - edge_loading_start).count();

        if (storage_ptrs_.node_embeddings != nullptr) {
            current_subgraph_state_->global_to_local_index_map_ =
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(true);
        } else if (storage_ptrs_.node_features != nullptr) {
            current_subgraph_state_->global_to_local_index_map_ =
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(true);
        }

        torch::Tensor mapped_edges;
        torch::Tensor mapped_edges_dst_sort;
        if (storage_ptrs_.edges->dim1_size_ == 3) {
            mapped_edges =
                torch::stack({current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                              current_subgraph_state_->all_in_memory_edges_.select(1, 1),
                              current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))})
                    .transpose(0, 1);
        } else if (storage_ptrs_.edges->dim1_size_ == 2) {
            // initializeInMemorySubGraph 프로파일링 (간소화)
            torch::Tensor src_mapped, dst_mapped;
            {
                PROFILE_BUCKET_MAPPING(index_select_time_us);
                INCREMENT_COUNTER(index_select_calls);
                auto src_indices = current_subgraph_state_->all_in_memory_edges_.select(1, 0);
                src_mapped = current_subgraph_state_->global_to_local_index_map_.index_select(0, src_indices);
                global_bucket_mapping_profiler.global_to_local_lookups.fetch_add(src_indices.size(0));
                
                auto dst_indices = current_subgraph_state_->all_in_memory_edges_.select(1, -1);
                dst_mapped = current_subgraph_state_->global_to_local_index_map_.index_select(0, dst_indices);
                global_bucket_mapping_profiler.global_to_local_lookups.fetch_add(dst_indices.size(0));
            }
            
            // Tensor stack (별도 스코프)
            {
                PROFILE_BUCKET_MAPPING(tensor_stack_time_us);
                mapped_edges = torch::stack({src_mapped, dst_mapped}).transpose(0, 1);
            }
        } else {
            // TODO use a function for logging errors and throwing expections
            SPDLOG_ERROR("Unexpected number of edge columns");
            std::runtime_error("Unexpected number of edge columns");
        }

        current_subgraph_state_->all_in_memory_mapped_edges_ = mapped_edges;

        // initializeInMemorySubGraph merge_sorted_edge_buckets 측정
        {
            PROFILE_BUCKET_MAPPING(merge_sorted_buckets_time_us);
            mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
            mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);
            mapped_edges = mapped_edges.to(torch::kInt64);
            mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);
        }

        if (current_subgraph_state_->in_memory_subgraph_ != nullptr) {
            current_subgraph_state_->in_memory_subgraph_ = nullptr;
        }

        current_subgraph_state_->in_memory_subgraph_ = std::make_shared<MariusGraph>(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory(), num_hash_maps);

        current_subgraph_state_->in_memory_partition_ids_ = new_in_mem_partition_ids;
        current_subgraph_state_->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
        current_subgraph_state_->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
        current_subgraph_state_->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;

        if (prefetch_) {
            if (hasSwap()) {
                // update next_subgraph_state_ in background
                getNextSubGraph();
            }
        }
    } else {
        // Either nothing buffered (in memory training) or eval and doing full graph evaluation
        current_subgraph_state_ = std::make_shared<InMemorySubgraphState>();

        bool should_sort = false;

        EdgeList src_sort;
        EdgeList dst_sort;
        if (storage_ptrs_.train_edges != nullptr) {
            src_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kInt64);
            if (storage_ptrs_.train_edges_dst_sort != nullptr) {
                dst_sort = storage_ptrs_.train_edges_dst_sort->range(0, storage_ptrs_.train_edges_dst_sort->getDim0()).to(torch::kInt64);
            } else {
                dst_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kInt64);
                should_sort = true;
            }
        } else {
            src_sort = storage_ptrs_.edges->range(0, storage_ptrs_.edges->getDim0()).to(torch::kInt64);
            dst_sort = storage_ptrs_.edges->range(0, storage_ptrs_.edges->getDim0()).to(torch::kInt64);
            should_sort = true;
        }

        if (should_sort) {
            src_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, 0))).to(torch::kInt64);
            dst_sort = dst_sort.index_select(0, torch::argsort(dst_sort.select(1, -1))).to(torch::kInt64);
        }

        current_subgraph_state_->in_memory_subgraph_ = std::make_shared<MariusGraph>(src_sort, dst_sort, getNumNodesInMemory(), num_hash_maps);
    }
}

void GraphModelStorage::updateInMemorySubGraph() {
    if (prefetch_) {
        // wait until the prefetching has been completed
        std::unique_lock lock(*subgraph_lock_);
        subgraph_cv_->wait(lock, [this] { return prefetch_complete_ == true; });
        // need to wait for the subgraph to be prefetched to perform the swap, otherwise the prefetched buffer_index_map may be incorrect
        performSwap();
        // free previous subgraph
        current_subgraph_state_->in_memory_subgraph_ = nullptr;
        current_subgraph_state_ = nullptr;

        current_subgraph_state_ = next_subgraph_state_;
        next_subgraph_state_ = nullptr;
        prefetch_complete_ = false;

        if (hasSwap()) {
            // update next_subgraph_state_ in background
            getNextSubGraph();
        }
    } else {
        std::pair<std::vector<int>, std::vector<int>> current_swap_ids = getNextSwapIds();
        performSwap();
        updateInMemorySubGraph_(current_subgraph_state_, current_swap_ids);
    }
}

void GraphModelStorage::getNextSubGraph() {
    std::pair<std::vector<int>, std::vector<int>> next_swap_ids = getNextSwapIds();
    next_subgraph_state_ = std::make_shared<InMemorySubgraphState>();
    next_subgraph_state_->in_memory_subgraph_ = nullptr;
    std::thread(&GraphModelStorage::updateInMemorySubGraph_, this, next_subgraph_state_, next_swap_ids).detach();
}

void GraphModelStorage::updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Initialize timing variables at function scope
    double bucket_prep_time = 0.0;
    double node_mapping_time = 0.0;  // Node Mapping 전체 시간
    double edge_loading_time = 0.0;
    int64_t total_edges_loaded = 0;
    
    if (prefetch_) {
        subgraph_lock_->lock();
    }

    std::vector<int> evict_partition_ids = std::get<0>(swap_ids);
    std::vector<int> admit_partition_ids = std::get<1>(swap_ids);

    torch::Tensor admit_ids_tensor = torch::tensor(admit_partition_ids, torch::kCPU);

    int buffer_size = current_subgraph_state_->in_memory_partition_ids_.size(0);
    int num_edge_buckets_in_mem = current_subgraph_state_->in_memory_edge_bucket_ids_.size(0);
    int num_partitions = getNumPartitions();
    int num_swap_partitions = evict_partition_ids.size();
    int num_remaining_partitions = buffer_size - num_swap_partitions;

    // PHASE 1: Bucket preparation timing starts
    auto bucket_prep_start = std::chrono::high_resolution_clock::now();
    
    // get edge buckets that will be kept in memory
    torch::Tensor keep_mask = torch::ones({num_edge_buckets_in_mem}, torch::kBool);
    auto accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_edge_bucket_ids_ = current_subgraph_state_->in_memory_edge_bucket_ids_.accessor<int64_t, 1>();

    // Check with OpenMP parallelization if each edge bucket is associated with partitions to be evicted
#pragma omp parallel for
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_id = accessor_in_memory_edge_bucket_ids_[i];
        int64_t src_partition = edge_bucket_id / num_partitions;  // Extract source partition ID
        int64_t dst_partition = edge_bucket_id % num_partitions;  // Extract destination partition ID

        // Mark for removal if edge bucket is associated with one of the partitions to be evicted
        for (int j = 0; j < num_swap_partitions; j++) {
            if (src_partition == evict_partition_ids[j] || dst_partition == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;  // Do not keep this edge bucket
            }
        }
    }

    torch::Tensor in_mem_edge_bucket_ids = current_subgraph_state_->in_memory_edge_bucket_ids_.masked_select(keep_mask);
    torch::Tensor in_mem_edge_bucket_sizes = current_subgraph_state_->in_memory_edge_bucket_sizes_.masked_select(keep_mask);
    torch::Tensor local_or_global_edge_bucket_starts = current_subgraph_state_->in_memory_edge_bucket_starts_.masked_select(keep_mask);

    // get new in memory partition ids
    keep_mask = torch::ones({buffer_size}, torch::kBool);
    accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_partition_ids_ = current_subgraph_state_->in_memory_partition_ids_.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        int64_t partition_id = accessor_in_memory_partition_ids_[i];

        for (int j = 0; j < num_swap_partitions; j++) {
            if (partition_id == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
                break;
            }
        }
    }

    torch::Tensor old_in_mem_partition_ids = current_subgraph_state_->in_memory_partition_ids_.masked_select(keep_mask);
    torch::Tensor new_in_mem_partition_ids = current_subgraph_state_->in_memory_partition_ids_.masked_scatter(~keep_mask, admit_ids_tensor);
    auto old_in_mem_partition_ids_accessor = old_in_mem_partition_ids.accessor<int64_t, 1>();
    auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

    // get new incoming edge buckets
    int num_new_edge_buckets = num_swap_partitions * (num_remaining_partitions + buffer_size);

    torch::Tensor new_edge_bucket_ids = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_edge_bucket_sizes = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_global_edge_bucket_starts = torch::zeros({num_new_edge_buckets}, torch::kInt64);

    auto new_edge_bucket_ids_accessor = new_edge_bucket_ids.accessor<int64_t, 1>();
    auto new_edge_bucket_sizes_accessor = new_edge_bucket_sizes.accessor<int64_t, 1>();
    auto new_global_edge_bucket_starts_accessor = new_global_edge_bucket_starts.accessor<int64_t, 1>();

    // TODO we don't need to do this every time
    std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
    torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int)edge_bucket_sizes_.size()}, torch::kInt64);
    torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
    torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
    auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
    auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < num_remaining_partitions; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = old_in_mem_partition_ids_accessor[i] * num_partitions + admit_partition_ids[j];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    int offset = num_swap_partitions * num_remaining_partitions;

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = admit_partition_ids[j] * num_partitions + new_in_mem_partition_ids_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = offset + i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    // concatenate old and new
    in_mem_edge_bucket_ids = torch::cat({in_mem_edge_bucket_ids, new_edge_bucket_ids});
    in_mem_edge_bucket_sizes = torch::cat({in_mem_edge_bucket_sizes, new_edge_bucket_sizes});
    local_or_global_edge_bucket_starts = torch::cat({local_or_global_edge_bucket_starts, new_global_edge_bucket_starts});

    torch::Tensor in_mem_mask = torch::ones({num_edge_buckets_in_mem - num_new_edge_buckets}, torch::kBool);
    in_mem_mask = torch::cat({in_mem_mask, torch::zeros({num_new_edge_buckets}, torch::kBool)});

    // put the ids in the correct order so the mapped edges remain sorted
    torch::Tensor src_ids_order = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
    auto src_ids_order_accessor = src_ids_order.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < buffer_size; j++) {
            int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];

            int idx = i * buffer_size + j;
            src_ids_order_accessor[idx] = edge_bucket_id;
        }
    }

    // TODO: all these argsorts can be done with one omp for loop, probably faster, same with masked_selects above
    torch::Tensor arg_sort = torch::argsort(in_mem_edge_bucket_ids);
    arg_sort = (arg_sort.index_select(0, torch::argsort(torch::argsort(src_ids_order))));
    in_mem_edge_bucket_ids = (in_mem_edge_bucket_ids.index_select(0, arg_sort));
    in_mem_edge_bucket_sizes = (in_mem_edge_bucket_sizes.index_select(0, arg_sort));
    local_or_global_edge_bucket_starts = (local_or_global_edge_bucket_starts.index_select(0, arg_sort));
    in_mem_mask = (in_mem_mask.index_select(0, arg_sort));
    
    auto bucket_prep_end = std::chrono::high_resolution_clock::now();
    bucket_prep_time = std::chrono::duration<double, std::milli>(bucket_prep_end - bucket_prep_start).count();

    // with everything in order grab the edge buckets
    torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
    int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
    in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

    auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
    auto local_or_global_edge_bucket_starts_accessor = local_or_global_edge_bucket_starts.accessor<int64_t, 1>();
    auto in_mem_mask_accessor = in_mem_mask.accessor<bool, 1>();
    auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

    torch::Tensor new_all_in_memory_edges = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

    // PHASE 2: Edge loading timing starts
    auto edge_loading_start = std::chrono::high_resolution_clock::now();
    
    // Collect edge data in parallel - load from existing memory or from disk
#pragma omp parallel for reduction(+:total_edges_loaded)
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
        int64_t edge_bucket_start = local_or_global_edge_bucket_starts_accessor[i];
        bool in_mem = in_mem_mask_accessor[i];      // Whether already in memory
        int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];
        
        total_edges_loaded += edge_bucket_size;

        if (in_mem) {
            // Copy edge bucket that is already in memory
            new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) =
                current_subgraph_state_->all_in_memory_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        } else {
            // Load new edge bucket from disk
            new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.edges->range(edge_bucket_start, edge_bucket_size);
        }
    }
    
    auto edge_loading_end = std::chrono::high_resolution_clock::now();
    edge_loading_time = std::chrono::duration<double, std::milli>(edge_loading_end - edge_loading_start).count();

    subgraph->all_in_memory_edges_ = new_all_in_memory_edges;

    if (storage_ptrs_.node_embeddings != nullptr) {
        subgraph->global_to_local_index_map_ =
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(!prefetch_);
    } else if (storage_ptrs_.node_features != nullptr) {
        subgraph->global_to_local_index_map_ = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(!prefetch_);
    }
    
    // PHASE 3: Bucket mapping timing starts - DETAILED PROFILING
    auto bucket_mapping_start = std::chrono::high_resolution_clock::now();
    
    // Don't reset profiler - we want to accumulate across all swaps in the epoch

    torch::Tensor mapped_edges;
    torch::Tensor mapped_edges_dst_sort;
    if (storage_ptrs_.edges->dim1_size_ == 3) {
        // Phase 3.1: Source node mapping
        torch::Tensor src_mapped;
        torch::Tensor dst_mapped;
        {
            PROFILE_BUCKET_MAPPING(index_select_time_us);
            INCREMENT_COUNTER(index_select_calls);
            auto src_indices = subgraph->all_in_memory_edges_.select(1, 0);
            src_mapped = subgraph->global_to_local_index_map_.index_select(0, src_indices);
            global_bucket_mapping_profiler.global_to_local_lookups.fetch_add(src_indices.size(0));
        }
        
        // Phase 3.2: Destination node mapping  
        {
            PROFILE_BUCKET_MAPPING(index_select_time_us);
            INCREMENT_COUNTER(index_select_calls);
            auto dst_indices = subgraph->all_in_memory_edges_.select(1, -1);
            dst_mapped = subgraph->global_to_local_index_map_.index_select(0, dst_indices);
            global_bucket_mapping_profiler.global_to_local_lookups.fetch_add(dst_indices.size(0));
        }
        
        // Phase 3.3: Tensor stack operation
        {
            PROFILE_BUCKET_MAPPING(tensor_stack_time_us);
            mapped_edges = torch::stack({src_mapped, 
                                       subgraph->all_in_memory_edges_.select(1, 1),
                                       dst_mapped}).transpose(0, 1);
        }
    } else if (storage_ptrs_.edges->dim1_size_ == 2) {
        // Simple profiling for Node Mapping operations
        torch::Tensor src_mapped;
        torch::Tensor dst_mapped;
        {
            PROFILE_BUCKET_MAPPING(index_select_time_us);
            INCREMENT_COUNTER(index_select_calls);
            auto src_indices = subgraph->all_in_memory_edges_.select(1, 0);
            src_mapped = subgraph->global_to_local_index_map_.index_select(0, src_indices);
            global_bucket_mapping_profiler.global_to_local_lookups.fetch_add(src_indices.size(0));
        }
        
        {
            PROFILE_BUCKET_MAPPING(index_select_time_us);
            INCREMENT_COUNTER(index_select_calls);
            auto dst_indices = subgraph->all_in_memory_edges_.select(1, -1);
            dst_mapped = subgraph->global_to_local_index_map_.index_select(0, dst_indices);
            global_bucket_mapping_profiler.global_to_local_lookups.fetch_add(dst_indices.size(0));
        }
        
        {
            PROFILE_BUCKET_MAPPING(tensor_stack_time_us);
            mapped_edges = torch::stack({src_mapped, dst_mapped}).transpose(0, 1);
        }
    } else {
        // TODO use a function for logging errors and throwing expections
        SPDLOG_ERROR("Unexpected number of edge columns");
        std::runtime_error("Unexpected number of edge columns");
    }

    //    assert((mapped_edges == -1).nonzero().size(0) == 0);
    //    assert((mapped_edges_dst_sort == -1).nonzero().size(0) == 0);

    subgraph->all_in_memory_mapped_edges_ = mapped_edges;

    // **실제 병목 지점**: merge_sorted_edge_buckets 측정
    {
        PROFILE_BUCKET_MAPPING(merge_sorted_buckets_time_us);
        mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
        mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);
        mapped_edges = mapped_edges.to(torch::kInt64);
        mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);
    }
    
    auto bucket_mapping_end = std::chrono::high_resolution_clock::now();
    node_mapping_time = std::chrono::duration<double, std::milli>(bucket_mapping_end - bucket_mapping_start).count();
    
    // Accumulate profiling data but don't print every swap to reduce log spam
    // Final summary will be printed at epoch end

    int num_hash_maps = current_subgraph_state_->in_memory_subgraph_->num_hash_maps_;

    if (subgraph->in_memory_subgraph_ != nullptr) {
        subgraph->in_memory_subgraph_ = nullptr;
    }

    subgraph->in_memory_subgraph_ = std::make_shared<MariusGraph>(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory(), num_hash_maps);

    // update state
    subgraph->in_memory_partition_ids_ = new_in_mem_partition_ids;
    subgraph->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
    subgraph->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
    subgraph->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;

    if (prefetch_) {
        prefetch_complete_ = true;
        subgraph_lock_->unlock();
        subgraph_cv_->notify_all();
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // Accumulate timing data (thread-safe)
    {
        std::lock_guard<std::mutex> lock(timing_mutex);
        timing_accumulator["bucket_preparation"] += bucket_prep_time;
        timing_accumulator["edge_bucket_metadata"] += 0.0;  // 항상 0으로 설정
        timing_accumulator["node_mapping"] += node_mapping_time;
        timing_accumulator["edge_loading"] += edge_loading_time;
        timing_accumulator["total_subgraph_update"] += total_time;
        timing_accumulator["total_edges_loaded"] += total_edges_loaded;
        
        timing_counter["bucket_preparation"]++;
        timing_counter["edge_bucket_metadata"]++;
        timing_counter["node_mapping"]++;
        timing_counter["edge_loading"]++;
        timing_counter["total_subgraph_update"]++;
        
        // Accumulate data only, no immediate logging to avoid cluttering output
        // Summary will be printed at epoch end via DataLoader::nextEpoch()
    }
}

// Get accumulated timing data for performance analysis
std::unordered_map<std::string, double> GraphModelStorage::getTimingData() {
    std::lock_guard<std::mutex> lock(timing_mutex);
    std::unordered_map<std::string, double> result;
    
    for (const auto& entry : timing_accumulator) {
        result[entry.first + "_total"] = entry.second;
    }
    
    for (const auto& entry : timing_counter) {
        result[entry.first + "_calls"] = entry.second;
        if (entry.second > 0 && timing_accumulator.find(entry.first) != timing_accumulator.end()) {
            result[entry.first + "_avg"] = timing_accumulator.at(entry.first) / entry.second;
        }
    }
    
    return result;
}

// Reset accumulated timing data (called at epoch start)
void GraphModelStorage::resetTimingData() {
    std::lock_guard<std::mutex> lock(timing_mutex);
    timing_accumulator.clear();
    timing_counter.clear();
}

// 정렬된 엣지 버킷들을 병합하는 메소드
// 소스 노드 또는 목적지 노드 기준으로 전체 엣지 리스트를 정렬
EdgeList GraphModelStorage::merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src) {
    int sort_dim = 0;   // 기본적으로 소스 노드(첫 번째 열) 기준 정렬
    if (!src) {
        sort_dim = -1;  // 목적지 노드(마지막 열) 기준 정렬
    }
    return edges.index_select(0, torch::argsort(edges.select(1, sort_dim)));  // 지정된 차원으로 정렬된 인덱스 반환
}

// 모든 엣지를 정렬하는 메소드 (학습/검증/테스트 엣지 포함)
// 인메모리 서브그래프 사용 여부에 따라 다른 정렬 전략 적용
void GraphModelStorage::sortAllEdges() {
    if (!useInMemorySubGraph()) {  // 전체 그래프가 메모리에 있는 경우
        std::vector<EdgeList> additional_edges = {};

        if (storage_ptrs_.train_edges != nullptr) {
            storage_ptrs_.train_edges->load();
            additional_edges.emplace_back(storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()));
        }

        if (storage_ptrs_.validation_edges != nullptr) {
            storage_ptrs_.validation_edges->load();
            additional_edges.emplace_back(storage_ptrs_.validation_edges->range(0, storage_ptrs_.validation_edges->getDim0()));
        }

        if (storage_ptrs_.test_edges != nullptr) {
            storage_ptrs_.test_edges->load();
            additional_edges.emplace_back(storage_ptrs_.test_edges->range(0, storage_ptrs_.test_edges->getDim0()));
        }

        for (auto f_edges : storage_ptrs_.filter_edges) {
            f_edges->load();
            additional_edges.emplace_back(f_edges->range(0, f_edges->getDim0()));
        }

        current_subgraph_state_->in_memory_subgraph_->sortAllEdges(torch::cat(additional_edges));

        for (auto f_edges : storage_ptrs_.filter_edges) {
            f_edges->unload();
        }
    } else {
        current_subgraph_state_->in_memory_subgraph_->sortAllEdges(current_subgraph_state_->in_memory_subgraph_->src_sorted_edges_);
    }
}