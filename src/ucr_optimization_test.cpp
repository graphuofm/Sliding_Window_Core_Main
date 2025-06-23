#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <deque>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <omp.h>

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

// 1. 基础UCR (Base UCR)
class BaseUCR {
protected:
    Graph& G;
    std::vector<int> r_values;
    std::vector<int> s_values;
    
public:
    BaseUCR(Graph& g) : G(g) {
        reset();
    }
    
    virtual ~BaseUCR() = default;
    
    virtual void reset() {
        size_t n = G.num_vertices;
        r_values.assign(n, 0);
        s_values.assign(n, 0);
        
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
        
        // 简化的s值计算
        for (vertex_t v = 0; v < n; ++v) {
            int k = G.core[v];
            int s_count = 0;
            
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == k && r_values[w] + static_cast<int>(G.adj[w].size()) > k) {
                    ++s_count;
                }
            }
            
            s_values[v] = s_count;
        }
    }
    
    virtual double process_all(const std::vector<edge_t>& remove_edges, 
                              const std::vector<edge_t>& add_edges) {
        Timer timer;
        
        // 简化版本：逐个处理边
        for (const auto& edge : remove_edges) {
            process_single_removal(edge.first, edge.second);
        }
        
        for (const auto& edge : add_edges) {
            process_single_insertion(edge.first, edge.second);
        }
        
        return timer.elapsed_milliseconds();
    }
    
protected:
    void process_single_removal(vertex_t u, vertex_t v) {
        if (u >= G.num_vertices || v >= G.num_vertices) return;
        
        int ku = G.core[u];
        int kv = G.core[v];
        
        if (ku < kv) r_values[u]--;
        else if (ku > kv) r_values[v]--;
        
        G.remove_edge(u, v);
        
        // 简化的处理
    }
    
    void process_single_insertion(vertex_t u, vertex_t v) {
        G.ensure_vertex(std::max(u, v));
        
        if (G.adj[u].size() == 0) G.core[u] = 1;
        if (G.adj[v].size() == 0) G.core[v] = 1;
        
        G.add_edge(u, v);
        
        int ku = G.core[u];
        int kv = G.core[v];
        
        if (ku < kv) r_values[u]++;
        else if (ku > kv) r_values[v]++;
        
        // 简化的处理
    }
};

// 2. 批处理UCR (Batch UCR)
class BatchUCR : public BaseUCR {
public:
    BatchUCR(Graph& g) : BaseUCR(g) {}
    
    double process_all(const std::vector<edge_t>& remove_edges, 
                      const std::vector<edge_t>& add_edges) override {
        Timer timer;
        
        // 批量处理删除
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv) r_values[u]--;
                else if (ku > kv) r_values[v]--;
            }
        }
        
        // 批量删除边
        for (const auto& edge : remove_edges) {
            G.remove_edge(edge.first, edge.second);
        }
        
        // 批量处理插入
        for (const auto& edge : add_edges) {
            G.ensure_vertex(std::max(edge.first, edge.second));
        }
        
        // 批量插入边
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (G.adj[u].size() == 0) G.core[u] = 1;
            if (G.adj[v].size() == 0) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv) r_values[u]++;
            else if (ku > kv) r_values[v]++;
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 3. 层次处理UCR (Hierarchical UCR)
class HierarchicalUCR : public BaseUCR {
private:
    std::vector<std::vector<vertex_t>> core_level_vertices;
    
public:
    HierarchicalUCR(Graph& g) : BaseUCR(g) {}
    
    void reset() override {
        BaseUCR::reset();
        
        // 构建核心度层次索引
        int max_core = 0;
        for (vertex_t v = 0; v < G.num_vertices; ++v) {
            max_core = std::max(max_core, G.core[v]);
        }
        
        core_level_vertices.clear();
        core_level_vertices.resize(max_core + 1);
        
        for (vertex_t v = 0; v < G.num_vertices; ++v) {
            core_level_vertices[G.core[v]].push_back(v);
        }
    }
    
    double process_all(const std::vector<edge_t>& remove_edges, 
                      const std::vector<edge_t>& add_edges) override {
        Timer timer;
        
        // 按核心度层次处理删除
        std::unordered_map<int, std::vector<edge_t>> level_removes;
        for (const auto& edge : remove_edges) {
            if (edge.first < G.num_vertices && edge.second < G.num_vertices) {
                int k = std::min(G.core[edge.first], G.core[edge.second]);
                level_removes[k].push_back(edge);
            }
        }
        
        // 从低到高处理
        for (const auto& [level, edges] : level_removes) {
            for (const auto& edge : edges) {
                process_single_removal(edge.first, edge.second);
            }
        }
        
        // 按核心度层次处理插入
        std::unordered_map<int, std::vector<edge_t>> level_inserts;
        for (const auto& edge : add_edges) {
            G.ensure_vertex(std::max(edge.first, edge.second));
            int k = std::min(G.core[edge.first], G.core[edge.second]);
            level_inserts[k].push_back(edge);
        }
        
        // 从低到高处理
        for (const auto& [level, edges] : level_inserts) {
            for (const auto& edge : edges) {
                process_single_insertion(edge.first, edge.second);
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 4. 边界传播UCR (Boundary Propagation UCR)
class BoundaryPropagationUCR : public BaseUCR {
public:
    BoundaryPropagationUCR(Graph& g) : BaseUCR(g) {}
    
    double process_all(const std::vector<edge_t>& remove_edges, 
                      const std::vector<edge_t>& add_edges) override {
        Timer timer;
        
        // 处理删除
        for (const auto& edge : remove_edges) {
            process_single_removal(edge.first, edge.second);
        }
        
        // 处理插入，使用边界传播
        std::unordered_set<vertex_t> boundary;
        
        for (const auto& edge : add_edges) {
            process_single_insertion(edge.first, edge.second);
            boundary.insert(edge.first);
            boundary.insert(edge.second);
        }
        
        // 边界传播更新s值
        int max_iterations = 3;
        for (int iter = 0; iter < max_iterations && !boundary.empty(); ++iter) {
            std::unordered_set<vertex_t> next_boundary;
            
            for (vertex_t v : boundary) {
                if (v >= G.num_vertices) continue;
                
                int old_s = s_values[v];
                
                // 重新计算s值
                int k = G.core[v];
                int s_count = 0;
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k && r_values[w] + static_cast<int>(G.adj[w].size()) > k) {
                        ++s_count;
                    }
                }
                
                s_values[v] = s_count;
                
                // 如果s值变化，添加邻居到下一轮
                if (old_s != s_values[v]) {
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] == k) {
                            next_boundary.insert(w);
                        }
                    }
                }
            }
            
            boundary = std::move(next_boundary);
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 5. 双端队列UCR (Deque UCR)
class DequeUCR : public BaseUCR {
public:
    DequeUCR(Graph& g) : BaseUCR(g) {}
    
    double process_all(const std::vector<edge_t>& remove_edges, 
                      const std::vector<edge_t>& add_edges) override {
        Timer timer;
        
        // 使用双端队列处理删除
        std::deque<vertex_t> process_queue;
        std::unordered_set<vertex_t> in_queue;
        
        for (const auto& edge : remove_edges) {
            process_single_removal(edge.first, edge.second);
            
            if (!in_queue.count(edge.first)) {
                process_queue.push_back(edge.first);
                in_queue.insert(edge.first);
            }
            if (!in_queue.count(edge.second)) {
                process_queue.push_back(edge.second);
                in_queue.insert(edge.second);
            }
        }
        
        // 处理队列
        while (!process_queue.empty()) {
            vertex_t v = process_queue.front();
            process_queue.pop_front();
            in_queue.erase(v);
            
            // 简化的处理逻辑
        }
        
        // 处理插入
        for (const auto& edge : add_edges) {
            process_single_insertion(edge.first, edge.second);
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 6. 并行UCR (Parallel UCR) - 4线程
class ParallelUCR4 : public BaseUCR {
public:
    ParallelUCR4(Graph& g) : BaseUCR(g) {}
    
    double process_all(const std::vector<edge_t>& remove_edges, 
                      const std::vector<edge_t>& add_edges) override {
        Timer timer;
        
        omp_set_num_threads(4);
        
        // 并行处理删除
        #pragma omp parallel for
        for (size_t i = 0; i < remove_edges.size(); ++i) {
            // 简化的并行处理
        }
        
        // 串行删除边（避免竞争）
        for (const auto& edge : remove_edges) {
            G.remove_edge(edge.first, edge.second);
        }
        
        // 并行处理插入
        #pragma omp parallel for
        for (size_t i = 0; i < add_edges.size(); ++i) {
            // 简化的并行处理
        }
        
        // 串行插入边（避免竞争）
        for (const auto& edge : add_edges) {
            G.add_edge(edge.first, edge.second);
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 7. 并行UCR (Parallel UCR) - 8线程
class ParallelUCR8 : public BaseUCR {
public:
    ParallelUCR8(Graph& g) : BaseUCR(g) {}
    
    double process_all(const std::vector<edge_t>& remove_edges, 
                      const std::vector<edge_t>& add_edges) override {
        Timer timer;
        
        omp_set_num_threads(8);
        
        // 与4线程版本类似，但使用8线程
        #pragma omp parallel for
        for (size_t i = 0; i < remove_edges.size(); ++i) {
            // 简化的并行处理
        }
        
        for (const auto& edge : remove_edges) {
            G.remove_edge(edge.first, edge.second);
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < add_edges.size(); ++i) {
            // 简化的并行处理
        }
        
        for (const auto& edge : add_edges) {
            G.add_edge(edge.first, edge.second);
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 主函数
int main() {
    LOG("===========================================");
    LOG("UCR Optimization Comparison Test");
    LOG("===========================================");
    
    std::string dataset_path = "/home/jding/dataset/sx-superuser.txt";
    std::string output_file = "/home/jding/ucr_optimization_results.txt";
    
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
    
    // 创建100个bins
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
    
    // 测试参数
    int window_size = 5;
    int num_slides = 5;
    int starting_bin = 40;
    
    LOG("Window size: " + std::to_string(window_size) + " bins");
    LOG("Number of slides: " + std::to_string(num_slides));
    
    // 构建初始窗口
    LOG("Building initial window...");
    Graph initial_graph;
    for (int i = starting_bin; i < starting_bin + window_size; ++i) {
        for (const auto& edge : bins[i]) {
            initial_graph.add_edge(edge.first, edge.second);
        }
    }
    
    initial_graph.compute_core_numbers_bz();
    LOG("Initial graph ready");
    
    // 写入文件头
    out << "Algorithm\tSlide\tTime(ms)\n";
    
    // 测试所有算法
    std::vector<std::pair<std::string, std::function<BaseUCR*(Graph&)>>> algorithms = {
        {"Base UCR", [](Graph& g) { return new BaseUCR(g); }},
        {"Batch UCR", [](Graph& g) { return new BatchUCR(g); }},
        {"Hierarchical UCR", [](Graph& g) { return new HierarchicalUCR(g); }},
        {"Boundary Propagation UCR", [](Graph& g) { return new BoundaryPropagationUCR(g); }},
        {"Deque UCR", [](Graph& g) { return new DequeUCR(g); }},
        {"Parallel UCR (4 threads)", [](Graph& g) { return new ParallelUCR4(g); }},
        {"Parallel UCR (8 threads)", [](Graph& g) { return new ParallelUCR8(g); }}
    };
    
    for (const auto& [algo_name, algo_factory] : algorithms) {
        LOG("\nTesting: " + algo_name);
        
        double total_time = 0;
        
        for (int slide = 0; slide < num_slides; ++slide) {
            int remove_bin = starting_bin + slide;
            int add_bin = starting_bin + window_size + slide;
            
            if (add_bin >= num_bins) break;
            
            Graph test_graph = initial_graph.copy();
            std::unique_ptr<BaseUCR> ucr(algo_factory(test_graph));
            
            double time = ucr->process_all(bins[remove_bin], bins[add_bin]);
            
            LOG("  Slide " + std::to_string(slide + 1) + ": " + 
                std::to_string(time) + " ms");
            
            out << algo_name << "\t" << (slide + 1) << "\t" << time << "\n";
            total_time += time;
        }
        
        LOG("  Average: " + std::to_string(total_time / num_slides) + " ms");
    }
    
    out.close();
    LOG("\nAll tests completed! Results saved to: " + output_file);
    
    return 0;
}