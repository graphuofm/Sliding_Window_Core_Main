#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

using vertex_t = int;
using edge_t = std::pair<vertex_t, vertex_t>;
using timestamp_t = long long;

struct TemporalEdge {
    vertex_t src, dst;
    timestamp_t timestamp;
    TemporalEdge(vertex_t s, vertex_t d, timestamp_t t) : src(s), dst(d), timestamp(t) {}
};

// 精确计时器
class PrecisionTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    
    double elapsed_microseconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end_time - start_time).count();
    }
};

// 简化的图结构
class Graph {
public:
    std::vector<std::vector<vertex_t>> adj;
    std::vector<int> core;
    
    void ensure_vertex(vertex_t v) {
        if (v >= adj.size()) {
            adj.resize(v + 1);
            core.resize(v + 1, 0);
        }
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        ensure_vertex(std::max(u, v));
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        if (u >= adj.size() || v >= adj.size()) return;
        
        auto it_u = std::find(adj[u].begin(), adj[u].end(), v);
        if (it_u != adj[u].end()) adj[u].erase(it_u);
        
        auto it_v = std::find(adj[v].begin(), adj[v].end(), u);
        if (it_v != adj[v].end()) adj[v].erase(it_v);
    }
    
    void compute_core_numbers() {
        if (adj.empty()) return;
        
        size_t n = adj.size();
        std::vector<int> degree(n);
        int max_degree = 0;
        
        for (vertex_t v = 0; v < n; ++v) {
            degree[v] = adj[v].size();
            max_degree = std::max(max_degree, degree[v]);
        }
        
        std::vector<std::vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < n; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        std::vector<bool> processed(n, false);
        std::fill(core.begin(), core.end(), 0);
        
        for (int d = 0; d <= max_degree; ++d) {
            for (vertex_t v : bins[d]) {
                if (processed[v]) continue;
                
                core[v] = d;
                processed[v] = true;
                
                for (vertex_t w : adj[v]) {
                    if (processed[w]) continue;
                    
                    if (degree[w] > d) {
                        auto& bin = bins[degree[w]];
                        auto it = std::find(bin.begin(), bin.end(), w);
                        if (it != bin.end()) bin.erase(it);
                        
                        --degree[w];
                        bins[degree[w]].push_back(w);
                    }
                }
            }
        }
    }
    
    size_t size() const { return adj.size(); }
};

// MCD/PCD算法实现（带详细计时）
class DetailedMCDPCDCore {
private:
    Graph& G;
    std::vector<int> MCD;
    std::vector<int> PCD;
    
    // 性能计时记录
    struct PerformanceMetrics {
        double mcd_computation_time = 0.0;
        double pcd_computation_time = 0.0;
        double mcd_cleanup_time = 0.0;
        double pcd_cleanup_time = 0.0;
        double graph_traversal_time = 0.0;
        double candidate_identification_time = 0.0;
        double core_update_time = 0.0;
        double memory_reallocation_time = 0.0;
        double total_time = 0.0;
    };
    
    PerformanceMetrics insert_metrics, remove_metrics;
    
    bool qual(vertex_t v, vertex_t w) const {
        if (v >= G.size() || w >= G.size()) return false;
        int k = G.core[v];
        return G.core[w] > k || (G.core[w] == k && MCD[w] > k);
    }
    
    void recompute_MCD(const std::unordered_set<vertex_t>& vertices, PrecisionTimer& timer) {
        timer.start();
        for (vertex_t v : vertices) {
            if (v >= G.size()) continue;
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] >= G.core[v]) {
                    ++count;
                }
            }
            MCD[v] = count;
        }
    }
    
    void recompute_PCD(const std::unordered_set<vertex_t>& vertices, PrecisionTimer& timer) {
        timer.start();
        for (vertex_t v : vertices) {
            if (v >= G.size()) continue;
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (qual(v, w)) {
                    ++count;
                }
            }
            PCD[v] = count;
        }
    }
    
public:
    DetailedMCDPCDCore(Graph& g) : G(g) {}
    
    void reset_and_compute(PrecisionTimer& timer) {
        timer.start();
        size_t n = G.size();
        
        // 内存重分配计时
        PrecisionTimer mem_timer;
        mem_timer.start();
        if (MCD.size() < n) {
            MCD.resize(n, 0);
            PCD.resize(n, 0);
        }
        double mem_time = mem_timer.elapsed_microseconds();
        
        // MCD计算计时
        PrecisionTimer mcd_timer;
        mcd_timer.start();
        for (vertex_t v = 0; v < n; ++v) {
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] >= G.core[v]) {
                    ++count;
                }
            }
            MCD[v] = count;
        }
        double mcd_time = mcd_timer.elapsed_microseconds();
        
        // PCD计算计时
        PrecisionTimer pcd_timer;
        pcd_timer.start();
        for (vertex_t v = 0; v < n; ++v) {
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (qual(v, w)) {
                    ++count;
                }
            }
            PCD[v] = count;
        }
        double pcd_time = pcd_timer.elapsed_microseconds();
        
        std::cout << "  - Memory reallocation: " << mem_time << " μs" << std::endl;
        std::cout << "  - MCD computation: " << mcd_time << " μs" << std::endl;
        std::cout << "  - PCD computation: " << pcd_time << " μs" << std::endl;
    }
    
    double process_edge_insertion(const std::vector<edge_t>& edges) {
        PrecisionTimer total_timer;
        total_timer.start();
        
        insert_metrics = PerformanceMetrics();
        
        // 1. 内存重分配检查
        PrecisionTimer mem_timer;
        mem_timer.start();
        vertex_t max_vertex = 0;
        for (const auto& edge : edges) {
            max_vertex = std::max(max_vertex, std::max(edge.first, edge.second));
        }
        if (max_vertex >= MCD.size()) {
            MCD.resize(max_vertex + 1, 0);
            PCD.resize(max_vertex + 1, 0);
        }
        insert_metrics.memory_reallocation_time = mem_timer.elapsed_microseconds();
        
        // 2. 添加边并更新MCD
        PrecisionTimer mcd_timer;
        mcd_timer.start();
        std::unordered_set<vertex_t> affected_vertices;
        for (const auto& edge : edges) {
            vertex_t u = edge.first, v = edge.second;
            G.add_edge(u, v);
            
            if (G.core[v] >= G.core[u]) ++MCD[u];
            if (G.core[u] >= G.core[v]) ++MCD[v];
            
            affected_vertices.insert(u);
            affected_vertices.insert(v);
            for (vertex_t w : G.adj[u]) affected_vertices.insert(w);
            for (vertex_t w : G.adj[v]) affected_vertices.insert(w);
        }
        insert_metrics.mcd_computation_time = mcd_timer.elapsed_microseconds();
        
        // 3. 重新计算PCD
        PrecisionTimer pcd_timer;
        pcd_timer.start();
        recompute_PCD(affected_vertices, pcd_timer);
        insert_metrics.pcd_computation_time = pcd_timer.elapsed_microseconds();
        
        // 4. 图遍历和候选识别
        PrecisionTimer traversal_timer;
        traversal_timer.start();
        std::unordered_set<vertex_t> candidates;
        for (vertex_t v : affected_vertices) {
            if (v < G.size() && PCD[v] > G.core[v]) {
                candidates.insert(v);
            }
        }
        insert_metrics.graph_traversal_time = traversal_timer.elapsed_microseconds();
        
        // 5. 核心度更新
        PrecisionTimer update_timer;
        update_timer.start();
        // 简化的核心度更新逻辑
        for (vertex_t v : candidates) {
            if (PCD[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        insert_metrics.core_update_time = update_timer.elapsed_microseconds();
        
        insert_metrics.total_time = total_timer.elapsed_microseconds();
        return insert_metrics.total_time;
    }
    
    double process_edge_removal(const std::vector<edge_t>& edges) {
        PrecisionTimer total_timer;
        total_timer.start();
        
        remove_metrics = PerformanceMetrics();
        
        // 1. MCD清理计时
        PrecisionTimer mcd_cleanup_timer;
        mcd_cleanup_timer.start();
        std::unordered_set<vertex_t> affected_vertices;
        for (const auto& edge : edges) {
            vertex_t u = edge.first, v = edge.second;
            if (u >= G.size() || v >= G.size()) continue;
            
            affected_vertices.insert(u);
            affected_vertices.insert(v);
            
            if (G.core[v] >= G.core[u]) --MCD[u];
            if (G.core[u] >= G.core[v]) --MCD[v];
            
            G.remove_edge(u, v);
        }
        remove_metrics.mcd_cleanup_time = mcd_cleanup_timer.elapsed_microseconds();
        
        // 2. PCD重新计算
        PrecisionTimer pcd_cleanup_timer;
        pcd_cleanup_timer.start();
        recompute_PCD(affected_vertices, pcd_cleanup_timer);
        remove_metrics.pcd_cleanup_time = pcd_cleanup_timer.elapsed_microseconds();
        
        // 3. 核心度降级处理
        PrecisionTimer update_timer;
        update_timer.start();
        for (vertex_t v : affected_vertices) {
            if (v < G.size() && MCD[v] < G.core[v]) {
                G.core[v]--;
            }
        }
        remove_metrics.core_update_time = update_timer.elapsed_microseconds();
        
        remove_metrics.total_time = total_timer.elapsed_microseconds();
        return remove_metrics.total_time;
    }
    
    void print_insert_metrics() const {
        std::cout << "  Insert Performance Breakdown:" << std::endl;
        std::cout << "    - Memory reallocation: " << insert_metrics.memory_reallocation_time << " μs" << std::endl;
        std::cout << "    - MCD computation: " << insert_metrics.mcd_computation_time << " μs" << std::endl;
        std::cout << "    - PCD computation: " << insert_metrics.pcd_computation_time << " μs" << std::endl;
        std::cout << "    - Graph traversal: " << insert_metrics.graph_traversal_time << " μs" << std::endl;
        std::cout << "    - Core update: " << insert_metrics.core_update_time << " μs" << std::endl;
        std::cout << "    - Total: " << insert_metrics.total_time << " μs" << std::endl;
    }
    
    void print_remove_metrics() const {
        std::cout << "  Remove Performance Breakdown:" << std::endl;
        std::cout << "    - MCD cleanup: " << remove_metrics.mcd_cleanup_time << " μs" << std::endl;
        std::cout << "    - PCD cleanup: " << remove_metrics.pcd_cleanup_time << " μs" << std::endl;
        std::cout << "    - Core update: " << remove_metrics.core_update_time << " μs" << std::endl;
        std::cout << "    - Total: " << remove_metrics.total_time << " μs" << std::endl;
    }
};

// 时序图处理器
class TemporalGraphProcessor {
private:
    std::vector<TemporalEdge> edges;
    std::vector<std::vector<edge_t>> bins;
    
public:
    TemporalGraphProcessor(const std::string& filepath) {
        load_data(filepath);
    }
    
    void load_data(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            std::istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts) {
                edges.emplace_back(src, dst, ts);
            }
        }
        
        std::cout << "Loaded " << edges.size() << " edges" << std::endl;
    }
    
    void create_bins(int num_bins) {
        if (edges.empty()) return;
        
        timestamp_t min_ts = edges[0].timestamp;
        timestamp_t max_ts = edges[0].timestamp;
        
        for (const auto& edge : edges) {
            min_ts = std::min(min_ts, edge.timestamp);
            max_ts = std::max(max_ts, edge.timestamp);
        }
        
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / num_bins;
        bins.resize(num_bins);
        
        for (const auto& edge : edges) {
            int bin_idx = std::min(static_cast<int>((edge.timestamp - min_ts) / bin_span), num_bins - 1);
            bins[bin_idx].push_back({edge.src, edge.dst});
        }
        
        std::cout << "Created " << num_bins << " bins" << std::endl;
    }
    
    const std::vector<edge_t>& get_bin_edges(int bin_idx) const {
        static const std::vector<edge_t> empty;
        if (bin_idx < 0 || bin_idx >= bins.size()) return empty;
        return bins[bin_idx];
    }
    
    size_t get_num_bins() const { return bins.size(); }
};

int main(int argc, char* argv[]) {
    std::string dataset = "dataset/sx-superuser.txt";  // 指定dataset目录
    int num_bins = 100;
    int window_size = 10;
    int num_slides = 10;
    
    if (argc > 1) {
        dataset = std::string("dataset/") + argv[1];
    }
    
    std::cout << "=== MCD/PCD Performance Bottleneck Analysis ===" << std::endl;
    std::cout << "Dataset: " << dataset << std::endl;
    std::cout << "Window size: " << window_size << " bins" << std::endl;
    std::cout << "Number of slides: " << num_slides << std::endl;
    std::cout << std::endl;
    
    // 加载数据
    TemporalGraphProcessor processor(dataset);
    processor.create_bins(num_bins);
    
    // 初始化图和算法
    Graph graph;
    DetailedMCDPCDCore mcd_pcd(graph);
    
    // 构建初始窗口
    std::cout << "Building initial window..." << std::endl;
    for (int i = 0; i < window_size; ++i) {
        const auto& bin_edges = processor.get_bin_edges(i);
        for (const auto& edge : bin_edges) {
            graph.add_edge(edge.first, edge.second);
        }
    }
    
    std::cout << "Computing initial core numbers..." << std::endl;
    PrecisionTimer init_timer;
    init_timer.start();
    graph.compute_core_numbers();
    double core_time = init_timer.elapsed_microseconds();
    std::cout << "Initial core computation: " << core_time << " μs" << std::endl;
    
    std::cout << "Initializing MCD/PCD..." << std::endl;
    mcd_pcd.reset_and_compute(init_timer);
    std::cout << std::endl;
    
    // 滑动窗口实验
    std::cout << "=== Sliding Window Analysis ===" << std::endl;
    double total_remove_time = 0.0;
    double total_insert_time = 0.0;
    
    for (int slide = 0; slide < num_slides; ++slide) {
        int remove_bin = slide;
        int add_bin = window_size + slide;
        
        if (add_bin >= processor.get_num_bins()) break;
        
        const auto& remove_edges = processor.get_bin_edges(remove_bin);
        const auto& add_edges = processor.get_bin_edges(add_bin);
        
        std::cout << "Slide " << (slide + 1) << ": Remove " << remove_edges.size() 
                  << " edges, Add " << add_edges.size() << " edges" << std::endl;
        
        // 边删除分析
        double remove_time = mcd_pcd.process_edge_removal(remove_edges);
        mcd_pcd.print_remove_metrics();
        
        // 边插入分析
        double insert_time = mcd_pcd.process_edge_insertion(add_edges);
        mcd_pcd.print_insert_metrics();
        
        total_remove_time += remove_time;
        total_insert_time += insert_time;
        
        std::cout << "Slide " << (slide + 1) << " total time: " 
                  << (remove_time + insert_time) << " μs" << std::endl;
        std::cout << std::endl;
    }
    
    // 总结
    std::cout << "=== Performance Summary ===" << std::endl;
    std::cout << "Average remove time: " << (total_remove_time / num_slides) << " μs" << std::endl;
    std::cout << "Average insert time: " << (total_insert_time / num_slides) << " μs" << std::endl;
    std::cout << "Total processing time: " << (total_remove_time + total_insert_time) << " μs" << std::endl;
    
    return 0;
}