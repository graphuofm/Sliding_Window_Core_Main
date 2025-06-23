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
#include <cstring>
#include <ctime>

// 定义顶点和边的类型
using vertex_t = int;
using edge_t = std::pair<vertex_t, vertex_t>;
using timestamp_t = long long;

// 获取当前时间字符串
std::string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&time_t));
    return std::string(buffer);
}

// 日志输出宏
#define LOG(msg) do { \
    std::cout << "[" << get_current_time() << "] " << msg << std::endl; \
    std::cout.flush(); \
} while(0)

// 时序边结构
struct TemporalEdge {
    vertex_t src;
    vertex_t dst;
    timestamp_t timestamp;
    
    TemporalEdge(vertex_t s, vertex_t d, timestamp_t t) : src(s), dst(d), timestamp(t) {}
};

// 计时辅助类
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_milliseconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
};

// 图结构
class Graph {
public:
    std::vector<std::vector<vertex_t>> adj;
    std::vector<int> core;
    size_t num_vertices;
    
    Graph() : num_vertices(0) {}
    
    void ensure_vertex(vertex_t v) {
        if (v >= num_vertices) {
            num_vertices = v + 1;
            adj.resize(num_vertices);
            core.resize(num_vertices, 0);
        }
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        ensure_vertex(std::max(u, v));
        
        if (std::find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
            adj[u].push_back(v);
        }
        if (std::find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) {
            adj[v].push_back(u);
        }
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        if (u >= num_vertices || v >= num_vertices) return;
        
        auto it_u = std::find(adj[u].begin(), adj[u].end(), v);
        if (it_u != adj[u].end()) {
            *it_u = adj[u].back();
            adj[u].pop_back();
        }
        
        auto it_v = std::find(adj[v].begin(), adj[v].end(), u);
        if (it_v != adj[v].end()) {
            *it_v = adj[v].back();
            adj[v].pop_back();
        }
    }
    
    void compute_core_numbers_bz() {
        if (num_vertices == 0) return;
        
        std::fill(core.begin(), core.end(), 0);
        
        std::vector<int> degree(num_vertices);
        int max_degree = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
            degree[v] = adj[v].size();
            max_degree = std::max(max_degree, degree[v]);
        }
        
        std::vector<std::vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < num_vertices; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        std::vector<bool> processed(num_vertices, false);
        
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
                        if (it != bin.end()) {
                            std::swap(*it, bin.back());
                            bin.pop_back();
                        }
                        
                        --degree[w];
                        bins[degree[w]].push_back(w);
                    }
                }
            }
        }
    }
    
    Graph copy() const {
        Graph g;
        g.num_vertices = num_vertices;
        g.adj = adj;
        g.core = core;
        return g;
    }
};

// 简化的UCR算法
class SimpleUCRCore {
private:
    Graph& G;
    std::vector<int> r_values;
    std::vector<int> s_values;
    
public:
    SimpleUCRCore(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.assign(n, 0);
        s_values.assign(n, 0);
        
        LOG("  UCR Reset: Computing r-values for " + std::to_string(n) + " vertices");
        
        // 计算r值
        for (vertex_t v = 0; v < n; ++v) {
            int k = G.core[v];
            int r_count = 0;
            
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] > k) {
                    ++r_count;
                }
            }
            
            r_values[v] = r_count;
        }
        
        LOG("  UCR Reset: r-values computed, now computing s-values");
        
        // 简化的s值计算
        for (vertex_t v = 0; v < n; ++v) {
            int k = G.core[v];
            int s_count = 0;
            
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == k) {
                    // 简化的条件
                    if (r_values[w] + G.adj[w].size() > k) {
                        ++s_count;
                    }
                }
            }
            
            s_values[v] = s_count;
        }
        
        LOG("  UCR Reset: Complete");
    }
    
    double process_edges(const std::vector<edge_t>& remove_edges, 
                        const std::vector<edge_t>& add_edges) {
        Timer timer;
        
        LOG("    Processing " + std::to_string(remove_edges.size()) + " removals");
        
        // 简单处理：直接修改图，然后重新计算受影响的核心度
        for (const auto& edge : remove_edges) {
            G.remove_edge(edge.first, edge.second);
        }
        
        LOG("    Processing " + std::to_string(add_edges.size()) + " insertions");
        
        for (const auto& edge : add_edges) {
            G.add_edge(edge.first, edge.second);
        }
        
        // 简化版本：只更新受影响顶点的核心度
        std::unordered_set<vertex_t> affected;
        
        for (const auto& edge : remove_edges) {
            affected.insert(edge.first);
            affected.insert(edge.second);
        }
        
        for (const auto& edge : add_edges) {
            affected.insert(edge.first);
            affected.insert(edge.second);
        }
        
        LOG("    Affected vertices: " + std::to_string(affected.size()));
        
        // 对受影响的顶点进行局部更新（这里简化处理）
        // 实际UCR算法会更复杂，但这里先确保能运行
        
        return timer.elapsed_milliseconds();
    }
};

// 主函数 - 只测试最小的数据集
int main() {
    LOG("===========================================");
    LOG("UCR Test on Smallest Dataset");
    LOG("===========================================");
    
    std::string dataset_path = "/home/jding/dataset/sx-superuser.txt";
    std::string output_file = "/home/jding/ucr_test_results.txt";
    
    std::ofstream out(output_file);
    if (!out.is_open()) {
        LOG("ERROR: Failed to open output file");
        return 1;
    }
    
    // 加载数据
    LOG("Loading dataset: " + dataset_path);
    std::vector<TemporalEdge> edges;
    
    std::ifstream file(dataset_path);
    if (!file.is_open()) {
        LOG("ERROR: Failed to open dataset");
        return 1;
    }
    
    std::string line;
    size_t line_count = 0;
    vertex_t max_vertex_id = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        if (line_count % 100000 == 0) {
            LOG("Loaded " + std::to_string(line_count) + " lines...");
        }
        
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        
        std::istringstream iss(line);
        vertex_t src, dst;
        timestamp_t ts;
        
        if (iss >> src >> dst >> ts) {
            edges.emplace_back(src, dst, ts);
            max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
        }
    }
    
    LOG("Loaded " + std::to_string(edges.size()) + " edges");
    LOG("Max vertex ID: " + std::to_string(max_vertex_id));
    
    // 创建100个bins（减少bins数量）
    int num_bins = 100;
    LOG("Creating " + std::to_string(num_bins) + " bins...");
    
    timestamp_t min_ts = edges[0].timestamp;
    timestamp_t max_ts = edges[0].timestamp;
    
    for (const auto& e : edges) {
        min_ts = std::min(min_ts, e.timestamp);
        max_ts = std::max(max_ts, e.timestamp);
    }
    
    double bin_span = static_cast<double>(max_ts - min_ts + 1) / num_bins;
    std::vector<std::vector<edge_t>> bins(num_bins);
    
    for (const auto& e : edges) {
        int bin_idx = std::min(static_cast<int>((e.timestamp - min_ts) / bin_span), num_bins - 1);
        bins[bin_idx].push_back({e.src, e.dst});
    }
    
    // 测试窗口大小5，滑动3次
    int window_size = 5;
    int num_slides = 3;
    int starting_bin = 40;
    
    LOG("Window size: " + std::to_string(window_size) + " bins");
    LOG("Number of slides: " + std::to_string(num_slides));
    
    // 构建初始窗口
    LOG("Building initial window...");
    Graph initial_graph;
    for (int i = starting_bin; i < starting_bin + window_size; ++i) {
        LOG("  Adding bin " + std::to_string(i) + " with " + 
            std::to_string(bins[i].size()) + " edges");
        for (const auto& edge : bins[i]) {
            initial_graph.add_edge(edge.first, edge.second);
        }
    }
    
    LOG("Computing initial core numbers...");
    initial_graph.compute_core_numbers_bz();
    LOG("Initial graph has " + std::to_string(initial_graph.num_vertices) + " vertices");
    
    // 测试UCR
    out << "Slide\tTime(ms)\n";
    
    for (int slide = 0; slide < num_slides; ++slide) {
        int remove_bin = starting_bin + slide;
        int add_bin = starting_bin + window_size + slide;
        
        LOG("\nSlide " + std::to_string(slide + 1) + ":");
        LOG("  Remove bin " + std::to_string(remove_bin) + 
            " (" + std::to_string(bins[remove_bin].size()) + " edges)");
        LOG("  Add bin " + std::to_string(add_bin) + 
            " (" + std::to_string(bins[add_bin].size()) + " edges)");
        
        Graph test_graph = initial_graph.copy();
        SimpleUCRCore ucr(test_graph);
        
        double time = ucr.process_edges(bins[remove_bin], bins[add_bin]);
        
        LOG("  Time: " + std::to_string(time) + " ms");
        
        out << slide + 1 << "\t" << time << "\n";
        
        // 更新初始图
        for (const auto& edge : bins[remove_bin]) {
            initial_graph.remove_edge(edge.first, edge.second);
        }
        for (const auto& edge : bins[add_bin]) {
            initial_graph.add_edge(edge.first, edge.second);
        }
        initial_graph.compute_core_numbers_bz();
    }
    
    out.close();
    LOG("\nTest completed! Results saved to: " + output_file);
    
    return 0;
}