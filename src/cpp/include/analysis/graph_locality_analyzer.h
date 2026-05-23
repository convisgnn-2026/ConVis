//
// Created for Smart-Dispersed GNN Research
// Graph Locality Analysis for understanding partition access patterns
//

#ifndef MARIUS_GRAPH_LOCALITY_ANALYZER_H
#define MARIUS_GRAPH_LOCALITY_ANALYZER_H

#include <unordered_map>
#include <vector>
#include <set>
#include "common/datatypes.h"
#include "reporting/logger.h"

struct PartitionAccessPattern {
    int partition_id;
    int64_t access_count;
    std::vector<int> co_accessed_partitions;  // Partitions accessed together
    double temporal_locality_score;           // How often accessed consecutively
    double spatial_locality_score;            // How many neighbors in same partition
};

struct GraphLocalityMetrics {
    // Community Structure Analysis
    double modularity_score;                  // Graph modularity (0-1)
    std::vector<std::vector<int>> communities; // Detected communities
    double partition_community_alignment;     // How well partitions align with communities
    
    // Cross-Partition Edge Analysis  
    int64_t total_edges;
    int64_t intra_partition_edges;            // Edges within same partition
    int64_t inter_partition_edges;            // Edges crossing partitions
    double edge_locality_ratio;               // intra_partition / total
    
    // Access Pattern Analysis
    std::vector<PartitionAccessPattern> partition_patterns;
    double average_temporal_locality;
    double average_spatial_locality;
    
    // Hot Partition Detection
    std::vector<int> hot_partitions;          // Top 20% most accessed
    std::vector<int> cold_partitions;         // Bottom 20% least accessed
    double access_skewness;                   // How skewed the access pattern is
    
    void print_analysis() {
        SPDLOG_INFO("=== GRAPH LOCALITY ANALYSIS ===");
        
        SPDLOG_INFO("Community Structure:");
        SPDLOG_INFO("  Modularity Score: {:.3f}", modularity_score);
        SPDLOG_INFO("  Number of Communities: {}", communities.size());
        SPDLOG_INFO("  Partition-Community Alignment: {:.3f}", partition_community_alignment);
        
        SPDLOG_INFO("Edge Locality:");
        SPDLOG_INFO("  Total Edges: {}", total_edges);
        SPDLOG_INFO("  Intra-Partition Edges: {} ({:.1f}%)", 
                   intra_partition_edges, 
                   (double)intra_partition_edges / total_edges * 100);
        SPDLOG_INFO("  Inter-Partition Edges: {} ({:.1f}%)", 
                   inter_partition_edges,
                   (double)inter_partition_edges / total_edges * 100);
        SPDLOG_INFO("  Edge Locality Ratio: {:.3f}", edge_locality_ratio);
        
        SPDLOG_INFO("Access Patterns:");
        SPDLOG_INFO("  Average Temporal Locality: {:.3f}", average_temporal_locality);
        SPDLOG_INFO("  Average Spatial Locality: {:.3f}", average_spatial_locality);
        SPDLOG_INFO("  Access Skewness: {:.3f}", access_skewness);
        
        SPDLOG_INFO("Hot/Cold Partitions:");
        SPDLOG_INFO("  Hot Partitions ({}): [{}]", hot_partitions.size(),
                   hot_partitions.size() > 0 ? std::to_string(hot_partitions[0]) + "..." : "none");
        SPDLOG_INFO("  Cold Partitions ({}): [{}]", cold_partitions.size(),
                   cold_partitions.size() > 0 ? std::to_string(cold_partitions[0]) + "..." : "none");
        
        SPDLOG_INFO("===================================");
    }
};

class GraphLocalityAnalyzer {
private:
    int num_partitions_;
    int64_t num_nodes_;
    
    // Access tracking
    std::vector<int64_t> partition_access_counts_;
    std::vector<std::vector<int>> partition_access_history_;
    std::unordered_map<int, std::set<int>> partition_co_access_map_;
    
    // Graph structure analysis
    torch::Tensor adjacency_matrix_;
    std::vector<std::vector<int>> partition_to_nodes_;
    
public:
    GraphLocalityAnalyzer(int num_partitions, int64_t num_nodes);
    
    // Track partition access patterns during training
    void recordPartitionAccess(const std::vector<int>& accessed_partitions);
    void recordNodeAccess(const torch::Tensor& node_ids);
    
    // Analyze graph structure
    void analyzeGraphStructure(const torch::Tensor& edges);
    std::vector<std::vector<int>> detectCommunities(const torch::Tensor& edges);
    double calculateModularity(const torch::Tensor& edges, 
                              const std::vector<std::vector<int>>& communities);
    
    // Calculate locality metrics
    double calculateTemporalLocality(int partition_id);
    double calculateSpatialLocality(int partition_id, const torch::Tensor& edges);
    double calculatePartitionCommunityAlignment(const std::vector<std::vector<int>>& communities);
    
    // Generate comprehensive analysis
    GraphLocalityMetrics generateLocalityMetrics(const torch::Tensor& edges);
    
    // Suggest optimization strategies
    std::vector<std::pair<int, int>> suggestPartitionMerges();
    std::vector<int> suggestHotPartitionCaching();
    std::vector<std::vector<int>> suggestPartitionClusters();
    
    void reset();
};

// Global analyzer instance
extern GraphLocalityAnalyzer* global_locality_analyzer;

// Convenience macros
#define RECORD_PARTITION_ACCESS(partitions) \
    if (global_locality_analyzer != nullptr) { \
        global_locality_analyzer->recordPartitionAccess(partitions); \
    }

#define RECORD_NODE_ACCESS(node_ids) \
    if (global_locality_analyzer != nullptr) { \
        global_locality_analyzer->recordNodeAccess(node_ids); \
    }

#endif // MARIUS_GRAPH_LOCALITY_ANALYZER_H