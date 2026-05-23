import numpy as np
from collections import defaultdict

from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner

import time
import torch  # isort:skip
import pymetis

def dataframe_to_tensor(df):
    return torch.tensor(df.to_numpy())

def partition_edges_convis(edges, num_nodes, num_partitions, edge_weights=None):
    """
    Returns: edges_torch, offsets, edge_weights, node_partitions, partition_score
        node_partitions:  [num_nodes] int64 — node→partition mapping
        partition_score:  [num_partitions] int64 — sum of node degrees per partition
                          (used downstream as critical-heavy cache score)
    """
    start = time.time()
    
    if not isinstance(edges, torch.Tensor):
        edges = torch.tensor(edges)
    
    degrees = torch.zeros(num_nodes, dtype=torch.int64)
    batch_size = 1000000
    
    for i in range(0, edges.size(0), batch_size):
        end_idx = min(i + batch_size, edges.size(0))
        batch_edges = edges[i:end_idx]
        
        degrees.scatter_add_(0, batch_edges[:, 0].to(torch.int64), 
                           torch.ones(batch_edges.size(0), dtype=torch.int64))
        degrees.scatter_add_(0, batch_edges[:, 1].to(torch.int64), 
                           torch.ones(batch_edges.size(0), dtype=torch.int64))
    
    partition_size = int(np.ceil(num_nodes / num_partitions))
    node_partitions = torch.div(torch.arange(num_nodes, dtype=torch.int64), partition_size, rounding_mode="trunc")
    node_partitions = torch.clamp(node_partitions, max=num_partitions - 1)
    
    partition_size = int(np.ceil(num_nodes / num_partitions))
    
    src_partitions = node_partitions[edges[:, 0].to(torch.int64)]
    dst_partitions = node_partitions[edges[:, 1].to(torch.int64)]
    
    _, dst_args = torch.sort(dst_partitions, stable=True)
    _, src_args = torch.sort(src_partitions[dst_args], stable=True)
    sort_order = dst_args[src_args]
    
    edges_torch = edges[sort_order]
    if edge_weights is not None:
        edge_weights = edge_weights[sort_order]
    
    edge_bucket_ids = torch.stack(
        [
            node_partitions[edges_torch[:, 0].to(torch.int64)],
            node_partitions[edges_torch[:, -1].to(torch.int64)],
        ],
        dim=1,
    )
    offsets = np.zeros([num_partitions, num_partitions], dtype=int)
    
    unique_src, num_source = torch.unique_consecutive(edge_bucket_ids[:, 0], return_counts=True)
    num_source_offsets = torch.cumsum(num_source, 0) - num_source
    
    curr_src_unique = 0
    for i in range(num_partitions):
        if curr_src_unique < unique_src.size(0) and unique_src[curr_src_unique] == i:
            offset = num_source_offsets[curr_src_unique]
            num_edges_for_src = num_source[curr_src_unique]
            dst_ids = edge_bucket_ids[offset : offset + num_edges_for_src, 1]
            
            unique_dst, num_dst = torch.unique_consecutive(dst_ids, return_counts=True)
            for ud, nd in zip(unique_dst, num_dst):
                offsets[i][ud] = nd.item()
            curr_src_unique += 1
    
    offsets = list(offsets.flatten())
    
    end = time.time()
    
    if edges_torch.size(0) > 1000000:
        sample_size = 100000
        sample_indices = torch.randperm(edges_torch.size(0))[:sample_size]
        edges_sample = edges_torch[sample_indices].numpy()
    else:
        edges_sample = edges_torch.numpy()
    
    partitions_cpu = node_partitions.numpy()
    
    if len(edges_sample) > 0:
        intra_edges = sum(1 for src, dst in edges_sample if partitions_cpu[src] == partitions_cpu[dst])
        inter_edges = len(edges_sample) - intra_edges
        
    partition_sizes = np.bincount(partitions_cpu, minlength=num_partitions)

    partition_score = torch.zeros(num_partitions, dtype=torch.int64)
    partition_score.scatter_add_(0, node_partitions, degrees)

    return edges_torch, offsets, edge_weights, node_partitions, partition_score

def partition_edges(edges, num_nodes, num_partitions, edge_weights=None):
    partition_size = int(np.ceil(num_nodes / num_partitions))

    src_partitions = torch.div(edges[:, 0], partition_size, rounding_mode="trunc")
    dst_partitions = torch.div(edges[:, -1], partition_size, rounding_mode="trunc")

    _, dst_args = torch.sort(dst_partitions, stable=True)
    _, src_args = torch.sort(src_partitions[dst_args], stable=True)
    sort_order = dst_args[src_args]

    edges = edges[sort_order]
    if edge_weights is not None:
        edge_weights = edge_weights[sort_order]

    edge_bucket_ids = torch.div(edges, partition_size, rounding_mode="trunc")
    offsets = np.zeros([num_partitions, num_partitions], dtype=int)
    unique_src, num_source = torch.unique_consecutive(edge_bucket_ids[:, 0], return_counts=True)

    num_source_offsets = torch.cumsum(num_source, 0) - num_source

    curr_src_unique = 0
    for i in range(num_partitions):
        if curr_src_unique < unique_src.size(0) and unique_src[curr_src_unique] == i:
            offset = num_source_offsets[curr_src_unique]
            num_edges = num_source[curr_src_unique]
            dst_ids = edge_bucket_ids[offset : offset + num_edges, -1]

            unique_dst, num_dst = torch.unique_consecutive(dst_ids, return_counts=True)

            offsets[i][unique_dst] = num_dst
            curr_src_unique += 1

    offsets = list(offsets.flatten())

    return edges, offsets, edge_weights


class TorchPartitioner(Partitioner):
    def __init__(self, partitioned_evaluation):
        super().__init__()

        self.partitioned_evaluation = partitioned_evaluation
        self.node_partitions = None
        self.partition_score = None

    def partition_edges(
        self, train_edges_tens, valid_edges_tens, test_edges_tens, num_nodes, num_partitions, edge_weights=None
    ):
        train_edge_weights, valid_edge_weights, test_edge_weights = None, None, None
        if edge_weights is not None:
            train_edge_weights, valid_edge_weights, test_edge_weights = (
                edge_weights[0],
                edge_weights[1],
                edge_weights[2],
            )

        (
            train_edges_tens,
            train_offsets,
            train_edge_weights,
            self.node_partitions,
            self.partition_score,
        ) = partition_edges(
            train_edges_tens, num_nodes, num_partitions, edge_weights=train_edge_weights
        )

        valid_offsets = None
        test_offsets = None

        if self.partitioned_evaluation:
            if valid_edges_tens is not None:
                valid_edges_tens, valid_offsets, valid_edge_weights, _, _ = partition_edges(
                    valid_edges_tens, num_nodes, num_partitions, edge_weights=valid_edge_weights
                )

            if test_edges_tens is not None:
                test_edges_tens, test_offsets, test_edge_weights, _, _ = partition_edges(
                    test_edges_tens, num_nodes, num_partitions, edge_weights=test_edge_weights
                )

        return (
            train_edges_tens,
            train_offsets,
            valid_edges_tens,
            valid_offsets,
            test_edges_tens,
            test_offsets,
            [train_edge_weights, valid_edge_weights, test_edge_weights],
        )
