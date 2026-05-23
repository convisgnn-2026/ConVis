//
// Created for Smart-Dispersed GNN Research
// Detailed profiling for Bucket Mapping performance analysis
//

#ifndef MARIUS_BUCKET_MAPPING_PROFILER_H
#define MARIUS_BUCKET_MAPPING_PROFILER_H

#include <chrono>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "common/datatypes.h"
#include "reporting/logger.h"

// Simplified profiling for Node Mapping operations
struct BucketMappingProfiler {
    // Core measurements only
    std::atomic<int64_t> index_select_calls{0};
    std::atomic<int64_t> index_select_time_us{0};
    std::atomic<int64_t> tensor_stack_time_us{0};
    std::atomic<int64_t> merge_sorted_buckets_time_us{0};   // 실제 병목 지점
    std::atomic<int64_t> global_to_local_lookups{0};
    
    void reset() {
        index_select_calls = 0;
        index_select_time_us = 0;
        tensor_stack_time_us = 0;
        merge_sorted_buckets_time_us = 0;
        global_to_local_lookups = 0;
    }
    
    void print_detailed_stats() {
        // 핵심 병목 지점만 간단히 출력
        double total_time_ms = (index_select_time_us.load() + tensor_stack_time_us.load() + merge_sorted_buckets_time_us.load()) / 1000.0;
        
        SPDLOG_INFO("NODE_MAPPING_BREAKDOWN: total={:.1f}ms, index_select={:.1f}ms({:.1f}%), tensor_stack={:.1f}ms({:.1f}%), merge_buckets={:.1f}ms({:.1f}%)", 
                   total_time_ms,
                   index_select_time_us.load() / 1000.0,
                   total_time_ms > 0 ? (index_select_time_us.load() / 1000.0) / total_time_ms * 100.0 : 0.0,
                   tensor_stack_time_us.load() / 1000.0, 
                   total_time_ms > 0 ? (tensor_stack_time_us.load() / 1000.0) / total_time_ms * 100.0 : 0.0,
                   merge_sorted_buckets_time_us.load() / 1000.0,
                   total_time_ms > 0 ? (merge_sorted_buckets_time_us.load() / 1000.0) / total_time_ms * 100.0 : 0.0);
    }
};

// Global profiler instance
extern BucketMappingProfiler global_bucket_mapping_profiler;

// RAII timer for automatic time measurement
class ScopedTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::atomic<int64_t>* target_counter_;
    
public:
    ScopedTimer(std::atomic<int64_t>* target) : target_counter_(target) {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        target_counter_->fetch_add(duration.count());
    }
};

// Convenience macros for profiling
#define PROFILE_BUCKET_MAPPING(counter) ScopedTimer timer(&global_bucket_mapping_profiler.counter)
#define INCREMENT_COUNTER(counter) global_bucket_mapping_profiler.counter.fetch_add(1)

#endif // MARIUS_BUCKET_MAPPING_PROFILER_H