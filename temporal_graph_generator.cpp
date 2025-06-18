#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <atomic>

struct DatasetConfig {
    std::string name;
    std::string filename;
    long long num_edges;
    int num_nodes;
    int time_span_years;
    int max_k_core;
    bool is_time_dense;  // true for time-dense, false for uniform
};

class TemporalGraphGenerator {
private:
    std::mt19937 gen;
    
public:
    TemporalGraphGenerator() : gen(std::chrono::steady_clock::now().time_since_epoch().count()) {}
    
    void generateDataset(const DatasetConfig& config) {
        std::cout << "\n=== Generating " << config.name << " ===" << std::endl;
        std::cout << "Target: " << config.num_edges << " edges, " 
                  << config.num_nodes << " nodes" << std::endl;
        std::cout << "Time span: " << config.time_span_years << " years, "
                  << "Max K-core: " << config.max_k_core << std::endl;
        std::cout << "Distribution: " << (config.is_time_dense ? "Time-dense" : "Uniform") << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Time parameters
        long long time_span_seconds = static_cast<long long>(config.time_span_years) * 365 * 24 * 3600;
        long long start_timestamp = 1280000000;  // Base timestamp (July 2010)
        
        // Degree tracking for K-core constraint
        std::vector<int> degrees(config.num_nodes, 0);
        std::vector<std::mutex> degree_mutexes(config.num_nodes);
        
        // Thread-local data
        int num_threads = omp_get_max_threads();
        long long edges_per_thread = config.num_edges / num_threads;
        
        std::vector<std::vector<std::string>> thread_data(num_threads);
        
        // Reserve memory for each thread
        for (int i = 0; i < num_threads; i++) {
            thread_data[i].reserve(edges_per_thread + 1000);
        }
        
        // Progress tracking
        std::atomic<long long> generated_edges(0);
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::mt19937 local_gen(gen() + thread_id);
            std::uniform_int_distribution<int> node_dist(0, config.num_nodes - 1);
            std::uniform_real_distribution<double> real_dist(0.0, 1.0);
            
            long long thread_edges = edges_per_thread;
            if (thread_id == num_threads - 1) {
                thread_edges += config.num_edges % num_threads;  // Last thread handles remainder
            }
            
            // Time range for this thread
            long long thread_time_start = start_timestamp + (time_span_seconds * thread_id) / num_threads;
            long long thread_time_end = start_timestamp + (time_span_seconds * (thread_id + 1)) / num_threads;
            
            for (long long edge = 0; edge < thread_edges; edge++) {
                // Generate timestamp based on distribution
                long long timestamp;
                if (config.is_time_dense) {
                    // Exponential growth: more edges towards the end
                    double progress = static_cast<double>(edge) / thread_edges;
                    double density_factor = std::pow(progress, 2.0);  // Quadratic growth
                    timestamp = thread_time_start + static_cast<long long>(
                        density_factor * (thread_time_end - thread_time_start));
                } else {
                    // Uniform distribution
                    std::uniform_int_distribution<long long> time_dist(thread_time_start, thread_time_end);
                    timestamp = time_dist(local_gen);
                }
                
                // Generate nodes with K-core constraint
                int u, v;
                int attempts = 0;
                do {
                    u = node_dist(local_gen);
                    v = node_dist(local_gen);
                    attempts++;
                    
                    // Avoid self-loops
                    if (u == v) continue;
                    
                    // Check K-core constraint (approximate, for performance)
                    bool valid = true;
                    if (attempts < 10) {  // Only check for first few attempts to avoid infinite loops
                        if (degrees[u] >= config.max_k_core * 0.9 || degrees[v] >= config.max_k_core * 0.9) {
                            valid = false;
                        }
                    }
                    
                    if (valid) break;
                } while (attempts < 20);
                
                // Update degrees (thread-safe)
                {
                    std::lock_guard<std::mutex> lock1(degree_mutexes[u]);
                    degrees[u]++;
                }
                {
                    std::lock_guard<std::mutex> lock2(degree_mutexes[v]);
                    degrees[v]++;
                }
                
                // Store edge data
                thread_data[thread_id].push_back(
                    std::to_string(u) + " " + std::to_string(v) + " " + std::to_string(timestamp)
                );
                
                // Update progress
                if (edge % 100000 == 0) {
                    generated_edges += 100000;
                    
                    #pragma omp critical
                    {
                        double progress = static_cast<double>(generated_edges) / config.num_edges * 100.0;
                        std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                                  << progress << "% (" << generated_edges << "/" << config.num_edges << " edges)"
                                  << std::flush;
                    }
                }
            }
        }
        
        std::cout << "\nGeneration completed. Writing to file..." << std::endl;
        
        // Collect all data and sort by timestamp
        std::vector<std::pair<long long, std::string>> all_edges;
        for (int i = 0; i < num_threads; i++) {
            for (const auto& edge_str : thread_data[i]) {
                // Extract timestamp for sorting
                size_t last_space = edge_str.find_last_of(' ');
                long long timestamp = std::stoll(edge_str.substr(last_space + 1));
                all_edges.emplace_back(timestamp, edge_str);
            }
        }
        
        std::cout << "Sorting edges by timestamp..." << std::endl;
        std::sort(all_edges.begin(), all_edges.end());
        
        // Write to file
        std::ofstream file(config.filename);
        if (!file) {
            std::cerr << "Error: Cannot create file " << config.filename << std::endl;
            return;
        }
        
        for (const auto& edge : all_edges) {
            file << edge.second << "\n";
        }
        file.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        // Calculate file size
        std::ifstream file_check(config.filename, std::ifstream::ate | std::ifstream::binary);
        long long file_size = file_check.tellg();
        file_check.close();
        
        std::cout << "\n✓ " << config.name << " completed!" << std::endl;
        std::cout << "  File: " << config.filename << std::endl;
        std::cout << "  Size: " << (file_size / (1024*1024)) << " MB" << std::endl;
        std::cout << "  Time: " << duration.count() << " seconds" << std::endl;
        
        // K-core statistics
        int max_degree = *std::max_element(degrees.begin(), degrees.end());
        std::cout << "  Max degree: " << max_degree << " (K-core constraint: " << config.max_k_core << ")" << std::endl;
    }
};

int main() {
    std::cout << "Temporal Graph Dataset Generator" << std::endl;
    std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads" << std::endl;
    
    TemporalGraphGenerator generator;
    
    // Dataset configurations
    std::vector<DatasetConfig> datasets = {
        {
            "Citation Network",
            "citation_network_10M.txt",
            10000000,      // 10M edges
            1000000,       // 1M nodes  
            1,             // 1 year
            100,           // K-core ≤ 100
            true           // Time-dense
        },
        {
            "Social Media Network", 
            "social_media_100M.txt",
            100000000,     // 100M edges
            5000000,       // 5M nodes
            5,             // 5 years
            1000,          // K-core ≤ 1000
            true           // Time-dense
        },
        {
            "Communication Network",
            "communication_1B.txt", 
            1000000000,    // 1B edges
            10000000,      // 10M nodes
            10,            // 10 years
            500,           // K-core ≤ 500 (relaxed for uniform distribution)
            false          // Uniform
        }
    };
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (const auto& config : datasets) {
        generator.generateDataset(config);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::minutes>(total_end - total_start);
    
    std::cout << "\n🎉 All datasets generated successfully!" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " minutes" << std::endl;
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "  - citation_network_10M.txt" << std::endl;
    std::cout << "  - social_media_100M.txt" << std::endl; 
    std::cout << "  - communication_1B.txt" << std::endl;
    
    return 0;
}