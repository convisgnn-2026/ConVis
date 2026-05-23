//
// Created for Smart-Dispersed GNN Research
// Theoretical Performance Model for DISPERSED vs SEQUENTIAL modes
//

#ifndef MARIUS_PERFORMANCE_MODEL_H
#define MARIUS_PERFORMANCE_MODEL_H

#include <vector>
#include <unordered_map>
#include "common/datatypes.h"
#include "reporting/logger.h"

// Core performance model parameters identified from empirical analysis
struct PerformanceModelParams {
    // System Configuration
    int num_partitions;
    int buffer_capacity;
    int64_t num_nodes;
    int64_t num_edges;
    
    // Hardware Characteristics
    double memory_bandwidth_gbps;        // Memory bandwidth (GB/s)
    double disk_bandwidth_gbps;          // Disk I/O bandwidth (GB/s)
    double cpu_cache_miss_penalty_us;    // Cache miss penalty (microseconds)
    double gpu_memory_bandwidth_gbps;    // GPU memory bandwidth (GB/s)
    
    // Graph Structure Parameters
    double edge_locality_ratio;          // Fraction of edges within partitions
    double access_skewness;              // Standard deviation / mean of partition access
    double temporal_locality_score;      // Temporal locality (0-1)
    double spatial_locality_score;       // Spatial locality (0-1)
    
    // Empirically Measured Constants
    double base_bucket_mapping_time_us;  // Base time for bucket mapping operation
    double index_select_overhead_factor; // Overhead factor for index_select operations
    double tensor_stack_overhead_us;     // Fixed overhead for tensor stack operations
    double partition_swap_overhead_us;   // Fixed overhead per partition swap
    
    PerformanceModelParams() {
        // Default values based on Papers100M analysis
        memory_bandwidth_gbps = 50.0;
        disk_bandwidth_gbps = 2.0;
        cpu_cache_miss_penalty_us = 100.0;
        gpu_memory_bandwidth_gbps = 500.0;
        
        // From empirical analysis
        base_bucket_mapping_time_us = 1000.0;
        index_select_overhead_factor = 2.0;
        tensor_stack_overhead_us = 50.0;
        partition_swap_overhead_us = 200.0;
        
        // Graph-dependent (to be measured)
        edge_locality_ratio = 0.6;
        access_skewness = 2.0;
        temporal_locality_score = 0.3;
        spatial_locality_score = 0.6;
    }
};

// Performance predictions for different modes
struct PerformancePrediction {
    // Core Metrics
    double predicted_throughput_nodes_per_sec;
    double predicted_epoch_time_seconds;
    double predicted_memory_usage_gb;
    
    // Breakdown by Component
    double bucket_mapping_time_ms;
    double edge_loading_time_ms;
    double partition_swap_time_ms;
    double memory_copy_time_ms;
    double cache_miss_penalty_ms;
    
    // Efficiency Metrics
    double cache_hit_rate;
    double memory_utilization;
    double parallelization_efficiency;
    
    // Scalability Metrics
    double scalability_factor;           // Performance degradation vs. ideal
    double memory_pressure_factor;       // Memory pressure (1.0 = optimal)
    
    void print_prediction(const std::string& mode_name) {
        SPDLOG_INFO("=== PERFORMANCE PREDICTION: {} ===", mode_name);
        
        SPDLOG_INFO("Core Performance:");
        SPDLOG_INFO("  Predicted Throughput: {:.1f} nodes/sec", predicted_throughput_nodes_per_sec);
        SPDLOG_INFO("  Predicted Epoch Time: {:.1f} seconds", predicted_epoch_time_seconds);
        SPDLOG_INFO("  Predicted Memory Usage: {:.1f} GB", predicted_memory_usage_gb);
        
        SPDLOG_INFO("Time Breakdown:");
        SPDLOG_INFO("  Bucket Mapping: {:.1f} ms", bucket_mapping_time_ms);
        SPDLOG_INFO("  Edge Loading: {:.1f} ms", edge_loading_time_ms);
        SPDLOG_INFO("  Partition Swaps: {:.1f} ms", partition_swap_time_ms);
        SPDLOG_INFO("  Memory Operations: {:.1f} ms", memory_copy_time_ms);
        SPDLOG_INFO("  Cache Miss Penalty: {:.1f} ms", cache_miss_penalty_ms);
        
        SPDLOG_INFO("Efficiency Metrics:");
        SPDLOG_INFO("  Cache Hit Rate: {:.1f}%", cache_hit_rate * 100);
        SPDLOG_INFO("  Memory Utilization: {:.1f}%", memory_utilization * 100);
        SPDLOG_INFO("  Parallelization Efficiency: {:.1f}%", parallelization_efficiency * 100);
        SPDLOG_INFO("  Scalability Factor: {:.3f}", scalability_factor);
        SPDLOG_INFO("  Memory Pressure: {:.3f}", memory_pressure_factor);
        
        SPDLOG_INFO("=====================================");
    }
};

class PerformanceModel {
private:
    PerformanceModelParams params_;
    
    // Model components
    double calculateBucketMappingTime(int num_swaps, int64_t edges_per_swap, bool dispersed_mode);
    double calculateCacheMissPenalty(int num_swaps, bool dispersed_mode);
    double calculateMemoryPressure(double memory_usage_gb, double available_memory_gb);
    double calculateParallelizationEfficiency(int num_threads, double workload_balance);
    
public:
    PerformanceModel();
    PerformanceModel(const PerformanceModelParams& params);
    
    // Core prediction methods
    PerformancePrediction predictSequentialMode();
    PerformancePrediction predictDispersedMode();
    PerformancePrediction predictSmartDispersedMode(double optimization_factor = 1.0);
    
    // Model calibration
    void calibrateFromEmpiricalData(const std::vector<double>& measured_times,
                                   const std::vector<std::string>& measurement_labels);
    void updateGraphStructureParams(double edge_locality, double access_skewness,
                                   double temporal_locality, double spatial_locality);
    
    // Analysis methods
    double calculateOptimalBufferSize();
    std::vector<int> suggestOptimalPartitioning();
    double predictOptimizationGain(const std::string& optimization_type);
    
    // Mathematical model equations
    double dispersedModeTimeModel(int num_partitions, int buffer_capacity, int64_t num_edges);
    double sequentialModeTimeModel(int num_partitions, int64_t num_edges);
    
    // Comparative analysis
    void compareModesAndPrintAnalysis();
    std::unordered_map<std::string, double> getOptimizationTargets();
    
    // Setters
    void setSystemParams(int num_partitions, int buffer_capacity, int64_t num_nodes, int64_t num_edges);
    void setHardwareParams(double memory_bw, double disk_bw, double cache_penalty);
    void setGraphParams(double edge_locality, double access_skewness);
};

// Mathematical formulas for performance modeling
namespace PerformanceFormulas {
    // Core time complexity formulas
    double bucketMappingComplexity(int num_swaps, int64_t nodes_per_swap, double cache_miss_rate);
    double memoryAccessComplexity(int64_t data_size_bytes, double bandwidth_gbps, double cache_hit_rate);
    double diskIOComplexity(int64_t data_size_bytes, double bandwidth_gbps, int num_operations);
    
    // Cache behavior models
    double predictCacheHitRate(int buffer_capacity, int num_partitions, double temporal_locality);
    double calculateWorkingSetSize(int buffer_capacity, int64_t partition_size_bytes);
    
    // Scalability models
    double linearScalingModel(double base_time, double dataset_scale_factor);
    double quadraticScalingModel(double base_time, double dataset_scale_factor, double complexity_factor);
    
    // Memory pressure models
    double memoryPressureEffect(double used_memory_gb, double available_memory_gb, double pressure_threshold);
}

#endif // MARIUS_PERFORMANCE_MODEL_H