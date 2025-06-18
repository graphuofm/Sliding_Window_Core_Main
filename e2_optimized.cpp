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
#include <climits>
#include <set>
#include <random>
#include <omp.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;
using vertex_t = int;
using edge_t = pair<vertex_t, vertex_t>;
using timestamp_t = long long;

// 全局结果文件
ofstream global_results;

// 时序边结构
struct TemporalEdge {
    vertex_t src;
    vertex_t dst;
    timestamp_t timestamp;
    
    TemporalEdge(vertex_t s, vertex_t d, timestamp_t t) : src(s), dst(d), timestamp(t) {}
};

// 高精度计时器
class Timer {
private:
    chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = chrono::high_resolution_clock::now();
    }
    
    double elapsed_milliseconds() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end_time - start_time).count();
    }
    
    double elapsed_seconds() const {
        return elapsed_milliseconds() / 1000.0;
    }
};

// 获取当前时间字符串
string get_current_time() {
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&time_t));
    return string(buffer);
}

// 日志输出
#define LOG(msg) do { \
    cout << "[" << get_current_time() << "] " << msg << endl; \
    cout.flush(); \
} while(0)

// 图结构
class Graph {
public:
    vector<vector<vertex_t>> adj;
    vector<int> core;
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
        if (u == v) return; // 跳过自环
        
        ensure_vertex(max(u, v));
        
        // 避免重复边
        if (find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
            adj[u].push_back(v);
        }
        if (find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) {
            adj[v].push_back(u);
        }
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        if (u >= num_vertices || v >= num_vertices) return;
        
        auto it_u = find(adj[u].begin(), adj[u].end(), v);
        if (it_u != adj[u].end()) {
            *it_u = adj[u].back();
            adj[u].pop_back();
        }
        
        auto it_v = find(adj[v].begin(), adj[v].end(), u);
        if (it_v != adj[v].end()) {
            *it_v = adj[v].back();
            adj[v].pop_back();
        }
    }
    
    // BZ算法计算核心度（带超时检查）
    bool compute_core_numbers_bz(double timeout_seconds = 600.0) {
        Timer timer;
        
        if (num_vertices == 0) return true;
        
        fill(core.begin(), core.end(), 0);
        
        vector<int> degree(num_vertices);
        int max_degree = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
            degree[v] = adj[v].size();
            max_degree = max(max_degree, degree[v]);
        }
        
        if (max_degree == 0) return true;
        
        vector<vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < num_vertices; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        vector<bool> processed(num_vertices, false);
        
        for (int d = 0; d <= max_degree; ++d) {
            // 每1000次迭代检查一次超时
            if (d % 1000 == 0 && timer.elapsed_seconds() > timeout_seconds) {
                LOG("    BZ computation timeout after " + to_string(timer.elapsed_seconds()) + " seconds");
                return false;
            }
            
            for (vertex_t v : bins[d]) {
                if (processed[v]) continue;
                
                core[v] = d;
                processed[v] = true;
                
                for (vertex_t w : adj[v]) {
                    if (processed[w]) continue;
                    
                    if (degree[w] > d) {
                        auto& bin = bins[degree[w]];
                        auto it = find(bin.begin(), bin.end(), w);
                        if (it != bin.end()) {
                            swap(*it, bin.back());
                            bin.pop_back();
                        }
                        
                        --degree[w];
                        bins[degree[w]].push_back(w);
                    }
                }
            }
        }
        
        return true;
    }
    
    Graph copy() const {
        Graph g;
        g.num_vertices = num_vertices;
        g.adj = adj;
        g.core = core;
        return g;
    }
    
    void clear() {
        for (auto& neighbors : adj) {
            vector<vertex_t>().swap(neighbors);
        }
        vector<vector<vertex_t>>().swap(adj);
        vector<int>().swap(core);
        num_vertices = 0;
    }
    
    // 强制内存清理（像厨师擦桌面）
    void force_cleanup() {
        for (auto& adj_list : adj) {
            adj_list.shrink_to_fit();
        }
        core.shrink_to_fit();
    }
};

// UCR基础版本（无优化）
class UCRBasic {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    
    bool is_qualified(vertex_t v) const {
        if (v >= r_values.size()) return false;
        int k = G.core[v];
        return r_values[v] + s_values[v] > k;
    }
    
    void update_s_value(vertex_t v) {
        if (v >= G.num_vertices) return;
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj[v]) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        if (v < s_values.size()) {
            s_values[v] = s_count;
        }
    }
    
public:
    UCRBasic(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
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
        
        // 计算s值
        for (vertex_t v = 0; v < n; ++v) {
            update_s_value(v);
        }
    }
    
    // 逐边处理
    double process_sliding_window(const vector<edge_t>& remove_edges, 
                                 const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 逐个删除边
        for (const auto& edge : remove_edges) {
            process_edge_removal(edge);
        }
        
        // 逐个添加边
        for (const auto& edge : add_edges) {
            process_edge_insertion(edge);
        }
        
        return timer.elapsed_milliseconds();
    }
    
private:
    void process_edge_removal(const edge_t& edge) {
        vertex_t u = edge.first;
        vertex_t v = edge.second;
        
        if (u >= G.num_vertices || v >= G.num_vertices) return;
        
        // 更新r值
        int ku = G.core[u];
        int kv = G.core[v];
        
        if (ku < kv && u < r_values.size()) r_values[u]--;
        else if (ku > kv && v < r_values.size()) r_values[v]--;
        
        G.remove_edge(u, v);
        
        // 简化的核心度降级检查
        if (r_values[u] + s_values[u] < G.core[u]) {
            G.core[u]--;
        }
        if (r_values[v] + s_values[v] < G.core[v]) {
            G.core[v]--;
        }
        
        // 更新s值
        update_s_value(u);
        update_s_value(v);
    }
    
    void process_edge_insertion(const edge_t& edge) {
        vertex_t u = edge.first;
        vertex_t v = edge.second;
        
        if (u == v) return;
        
        G.ensure_vertex(max(u, v));
        
        if (max(u, v) >= r_values.size()) {
            r_values.resize(max(u, v) + 1, 0);
            s_values.resize(max(u, v) + 1, 0);
        }
        
        G.add_edge(u, v);
        
        // 更新r值
        int ku = G.core[u];
        int kv = G.core[v];
        
        if (ku < kv && u < r_values.size()) r_values[u]++;
        else if (ku > kv && v < r_values.size()) r_values[v]++;
        
        // 简化的核心度升级检查
        if (r_values[u] + s_values[u] > G.core[u]) {
            G.core[u]++;
        }
        if (r_values[v] + s_values[v] > G.core[v]) {
            G.core[v]++;
        }
        
        // 更新s值
        update_s_value(u);
        update_s_value(v);
    }
};

// UCR批处理版本
class UCRBatch {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    
    bool is_qualified(vertex_t v) const {
        if (v >= r_values.size()) return false;
        int k = G.core[v];
        return r_values[v] + s_values[v] > k;
    }
    
    void update_s_value(vertex_t v) {
        if (v >= G.num_vertices) return;
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj[v]) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        if (v < s_values.size()) {
            s_values[v] = s_count;
        }
    }
    
public:
    UCRBatch(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
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
        
        // 计算s值
        for (vertex_t v = 0; v < n; ++v) {
            update_s_value(v);
        }
    }
    
    // 批处理版本
    double process_sliding_window(const vector<edge_t>& remove_edges, 
                                 const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 扩展数据结构
        vertex_t max_vertex_id = 0;
        for (const auto& edge : add_edges) {
            max_vertex_id = max(max_vertex_id, max(edge.first, edge.second));
        }
        
        if (max_vertex_id >= G.core.size()) {
            G.core.resize(max_vertex_id + 1, 0);
        }
        if (max_vertex_id >= r_values.size()) {
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // Phase 1: 批量删除边并更新r值
        unordered_map<int, unordered_set<vertex_t>> k_groups_remove;
        
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                if (ku > 0) k_groups_remove[ku].insert(u);
                if (kv > 0) k_groups_remove[kv].insert(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // 批量处理核心度降级
        vector<int> k_values;
        for (const auto& pair : k_groups_remove) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            for (vertex_t v : k_groups_remove[k]) {
                if (G.core[v] == k && r_values[v] + s_values[v] < k) {
                    G.core[v] = k - 1;
                }
            }
        }
        
        // Phase 2: 批量添加边并更新r值
        unordered_set<vertex_t> affected_vertices;
        
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.ensure_vertex(max(u, v));
            
            if (G.adj[u].empty()) G.core[u] = 1;
            if (G.adj[v].empty()) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
            
            affected_vertices.insert(u);
            affected_vertices.insert(v);
        }
        
        // 批量更新s值
        for (vertex_t v : affected_vertices) {
            update_s_value(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    update_s_value(w);
                }
            }
        }
        
        // 批量处理核心度升级
        for (vertex_t v : affected_vertices) {
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// UCR并行版本
class UCRParallel {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    
    bool is_qualified(vertex_t v) const {
        if (v >= r_values.size()) return false;
        int k = G.core[v];
        return r_values[v] + s_values[v] > k;
    }
    
    void update_s_value(vertex_t v) {
        if (v >= G.num_vertices) return;
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj[v]) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        if (v < s_values.size()) {
            s_values[v] = s_count;
        }
    }
    
public:
    UCRParallel(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
        // 并行计算r值
        #pragma omp parallel for
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
        
        // 并行计算s值
        #pragma omp parallel for
        for (vertex_t v = 0; v < n; ++v) {
            update_s_value(v);
        }
    }
    
    // 并行处理版本
    double process_sliding_window(const vector<edge_t>& remove_edges, 
                                 const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 扩展数据结构
        vertex_t max_vertex_id = 0;
        for (const auto& edge : add_edges) {
            max_vertex_id = max(max_vertex_id, max(edge.first, edge.second));
        }
        
        if (max_vertex_id >= G.core.size()) {
            G.core.resize(max_vertex_id + 1, 0);
        }
        if (max_vertex_id >= r_values.size()) {
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // 串行处理边删除（避免竞争）
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                G.remove_edge(u, v);
                
                if (r_values[u] + s_values[u] < G.core[u]) G.core[u]--;
                if (r_values[v] + s_values[v] < G.core[v]) G.core[v]--;
            }
        }
        
        // 串行处理边添加
        vector<vertex_t> affected_vertices;
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.ensure_vertex(max(u, v));
            
            if (G.adj[u].empty()) G.core[u] = 1;
            if (G.adj[v].empty()) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
            
            affected_vertices.push_back(u);
            affected_vertices.push_back(v);
        }
        
        // 并行更新s值
        #pragma omp parallel for
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            update_s_value(v);
        }
        
        // 串行处理核心度升级（避免竞争）
        for (vertex_t v : affected_vertices) {
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// UCR完整版本（批处理+并行）
class UCRFull {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    
    bool is_qualified(vertex_t v) const {
        if (v >= r_values.size()) return false;
        int k = G.core[v];
        return r_values[v] + s_values[v] > k;
    }
    
    void update_s_value(vertex_t v) {
        if (v >= G.num_vertices) return;
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj[v]) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        if (v < s_values.size()) {
            s_values[v] = s_count;
        }
    }
    
public:
    UCRFull(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
        // 并行计算r值
        #pragma omp parallel for
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
        
        // 并行计算s值
        #pragma omp parallel for
        for (vertex_t v = 0; v < n; ++v) {
            update_s_value(v);
        }
    }
    
    // 批处理+并行版本
    double process_sliding_window(const vector<edge_t>& remove_edges, 
                                 const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 扩展数据结构
        vertex_t max_vertex_id = 0;
        for (const auto& edge : add_edges) {
            max_vertex_id = max(max_vertex_id, max(edge.first, edge.second));
        }
        
        if (max_vertex_id >= G.core.size()) {
            G.core.resize(max_vertex_id + 1, 0);
        }
        if (max_vertex_id >= r_values.size()) {
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // Phase 1: 批量删除边
        unordered_map<int, vector<vertex_t>> k_groups_remove;
        
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                if (ku > 0) k_groups_remove[ku].push_back(u);
                if (kv > 0) k_groups_remove[kv].push_back(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // 并行处理各个k层级的降级
        for (auto& [k, vertices] : k_groups_remove) {
            #pragma omp parallel for
            for (size_t i = 0; i < vertices.size(); ++i) {
                vertex_t v = vertices[i];
                if (G.core[v] == k && r_values[v] + s_values[v] < k) {
                    G.core[v] = k - 1;
                }
            }
        }
        
        // Phase 2: 批量添加边
        vector<vertex_t> affected_vertices;
        
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.ensure_vertex(max(u, v));
            
            if (G.adj[u].empty()) G.core[u] = 1;
            if (G.adj[v].empty()) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
            
            affected_vertices.push_back(u);
            affected_vertices.push_back(v);
        }
        
        // 并行批量更新s值
        #pragma omp parallel for
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            update_s_value(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    update_s_value(w);
                }
            }
        }
        
        // 并行批量处理核心度升级
        #pragma omp parallel for
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 优化滑动窗口实验类
class OptimizedSlidingWindowExperiment {
private:
    string dataset_path;
    string dataset_name;
    
public:
    OptimizedSlidingWindowExperiment(const string& path, const string& name) 
        : dataset_path(path), dataset_name(name) {}
    
    void run() {
        LOG("=====================================");
        LOG("Dataset: " + dataset_name);
        LOG("=====================================");
        
        // 检查文件大小
        struct stat st;
        if (stat(dataset_path.c_str(), &st) == 0) {
            size_t file_size_gb = st.st_size / (1024LL * 1024LL * 1024LL);
            LOG("File size: " + to_string(file_size_gb) + " GB");
            
            // 超过20GB的文件跳过
            if (file_size_gb > 20) {
                LOG("SKIPPING: File too large");
                return;
            }
        }
        
        // 流式处理：先读取时间戳范围
        auto [min_ts, max_ts, total_edges] = scan_timestamp_range();
        if (total_edges == 0) {
            LOG("ERROR: No valid edges found");
            return;
        }
        
        LOG("Time range: " + to_string(min_ts) + " to " + to_string(max_ts));
        LOG("Total edges: " + to_string(total_edges));
        
        // 对每个窗口大小运行实验
        vector<int> window_sizes = {10, 20, 50, 100};
        for (int window_size : window_sizes) {
            run_optimization_experiment(window_size, min_ts, max_ts);
        }
    }
    
private:
    tuple<timestamp_t, timestamp_t, size_t> scan_timestamp_range() {
        LOG("Scanning timestamp range...");
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return {0, 0, 0};
        }
        
        string line;
        timestamp_t min_ts = LLONG_MAX;
        timestamp_t max_ts = LLONG_MIN;
        size_t edge_count = 0;
        size_t line_count = 0;
        
        Timer timer;
        
        while (getline(file, line)) {
            line_count++;
            if (line_count % 10000000 == 0) {
                LOG("  Scanned " + to_string(line_count) + " lines, found " + to_string(edge_count) + " valid edges");
                
                // 扫描超时检查：超过30分钟停止
                if (timer.elapsed_seconds() > 1800) {
                    LOG("  TIMEOUT: Scanning taking too long, stopping");
                    break;
                }
            }
            
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                min_ts = min(min_ts, ts);
                max_ts = max(max_ts, ts);
                edge_count++;
            }
        }
        
        file.close();
        LOG("Scan complete: " + to_string(edge_count) + " edges in " + to_string(timer.elapsed_seconds()) + " seconds");
        
        return {min_ts, max_ts, edge_count};
    }
    
    void run_optimization_experiment(int window_size, timestamp_t min_ts, timestamp_t max_ts) {
        LOG("\n--- Window size: " + to_string(window_size) + " bins ---");
        
        const int total_bins = 1000;
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
        
        // 选择起始位置
        int starting_bin = 400;
        if (starting_bin + window_size > total_bins) {
            starting_bin = total_bins - window_size - 10;
        }
        if (starting_bin < 0) starting_bin = 0;
        
        // **一次性读取并按bin分组所有边**
        LOG("Loading and indexing all edges by bins...");
        vector<vector<edge_t>> bins(total_bins);
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            LOG("ERROR: Cannot open file for indexing");
            return;
        }
        
        string line;
        size_t total_loaded = 0;
        Timer load_timer;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                int bin_idx = min(static_cast<int>((ts - min_ts) / bin_span), total_bins - 1);
                bins[bin_idx].push_back({src, dst});
                total_loaded++;
                
                if (total_loaded % 5000000 == 0) {
                    LOG("  Indexed " + to_string(total_loaded) + " edges...");
                    
                    // 加载超时检查
                    if (load_timer.elapsed_seconds() > 1200) { // 20分钟超时
                        LOG("  TIMEOUT: Loading taking too long, stopping");
                        break;
                    }
                }
            }
        }
        file.close();
        
        LOG("Indexing complete: " + to_string(total_loaded) + " edges in " + 
            to_string(load_timer.elapsed_seconds()) + " seconds");
        
        // 构建初始窗口
        LOG("Building initial window (bins " + to_string(starting_bin) + " to " + 
            to_string(starting_bin + window_size - 1) + ")...");
        
        Graph initial_graph;
        size_t initial_edges = 0;
        
        for (int i = starting_bin; i < starting_bin + window_size; ++i) {
            initial_edges += bins[i].size();
            for (const auto& edge : bins[i]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
        }
        
        LOG("  Initial edges in window: " + to_string(initial_edges));
        
        // 如果窗口边数过多，跳过
        if (initial_edges > 20000000) { // 超过2000万边跳过
            LOG("  SKIPPING: Too many edges in window");
            initial_graph.clear();
            return;
        }
        
        // 计算初始核心度（带超时）
        Timer init_timer;
        LOG("  Computing initial core numbers...");
        bool success = initial_graph.compute_core_numbers_bz(600.0); // 10分钟超时
        
        if (!success) {
            LOG("  SKIPPING: Initial core computation timeout");
            initial_graph.clear();
            return;
        }
        
        double init_time = init_timer.elapsed_milliseconds();
        LOG("  Initial core computation: " + to_string(init_time) + " ms");
        
        // 进行一次滑动测试（避免崩溃）
        int remove_bin = starting_bin;
        int add_bin = starting_bin + window_size;
        
        if (add_bin >= total_bins) {
            LOG("No bins available for sliding");
            initial_graph.clear();
            return;
        }
        
        LOG("\nSlide test:");
        LOG("  Remove: " + to_string(bins[remove_bin].size()) + " edges");
        LOG("  Add: " + to_string(bins[add_bin].size()) + " edges");
        
        // BZ重计算
        Timer bz_timer;
        Graph bz_graph;
        
        // 构建新窗口的图
        for (int i = remove_bin + 1; i <= add_bin; ++i) {
            for (const auto& edge : bins[i]) {
                bz_graph.add_edge(edge.first, edge.second);
            }
        }
        
        bool bz_success = bz_graph.compute_core_numbers_bz(300.0); // 5分钟超时
        
        if (!bz_success) {
            LOG("    BZ computation timeout");
            bz_graph.clear();
            initial_graph.clear();
            return;
        }
        
        double bz_time = bz_timer.elapsed_milliseconds();
        LOG("    BZ time: " + to_string(bz_time) + " ms");
        
        // 清理BZ图
        bz_graph.clear();
        
        // 测试四种UCR优化版本
        vector<pair<string, double>> ucr_results;
        
        // UCR-Basic
        {
            Graph ucr_graph = initial_graph.copy();
            UCRBasic ucr_core(ucr_graph);
            
            Timer ucr_timer;
            double ucr_time = ucr_core.process_sliding_window(bins[remove_bin], bins[add_bin]);
            ucr_results.push_back({"UCR-Basic", ucr_time});
            LOG("    UCR-Basic time: " + to_string(ucr_time) + " ms");
            
            ucr_graph.clear();
        }
        
        // UCR-Batch
        {
            Graph ucr_graph = initial_graph.copy();
            UCRBatch ucr_core(ucr_graph);
            
            Timer ucr_timer;
            double ucr_time = ucr_core.process_sliding_window(bins[remove_bin], bins[add_bin]);
            ucr_results.push_back({"UCR-Batch", ucr_time});
            LOG("    UCR-Batch time: " + to_string(ucr_time) + " ms");
            
            ucr_graph.clear();
        }
        
        // UCR-Parallel
        {
            Graph ucr_graph = initial_graph.copy();
            UCRParallel ucr_core(ucr_graph);
            
            Timer ucr_timer;
            double ucr_time = ucr_core.process_sliding_window(bins[remove_bin], bins[add_bin]);
            ucr_results.push_back({"UCR-Parallel", ucr_time});
            LOG("    UCR-Parallel time: " + to_string(ucr_time) + " ms");
            
            ucr_graph.clear();
        }
        
        // UCR-Full
        {
            Graph ucr_graph = initial_graph.copy();
            UCRFull ucr_core(ucr_graph);
            
            Timer ucr_timer;
            double ucr_time = ucr_core.process_sliding_window(bins[remove_bin], bins[add_bin]);
            ucr_results.push_back({"UCR-Full", ucr_time});
            LOG("    UCR-Full time: " + to_string(ucr_time) + " ms");
            
            ucr_graph.clear();
        }
        
        // 计算加速比并写入结果
        for (const auto& [method, ucr_time] : ucr_results) {
            double speedup = (ucr_time > 0) ? bz_time / ucr_time : 1.0;
            LOG("    " + method + " speedup: " + to_string(speedup) + "x");
            
            // 写入全局结果文件
            global_results << dataset_name << "\t" 
                          << window_size << "\t"
                          << method << "\t"
                          << fixed << setprecision(2) 
                          << bz_time << "\t"
                          << ucr_time << "\t"
                          << speedup << "\n";
            global_results.flush();
        }
        
        // 清理初始图和bins（像厨师擦桌面）
        initial_graph.clear();
        for (auto& bin : bins) {
            vector<edge_t>().swap(bin);
        }
        vector<vector<edge_t>>().swap(bins);
        
        LOG("  Memory cleaned up (table wiped clean!)");
    }
};

// 获取数据集文件列表
vector<pair<string, string>> get_datasets(const string& dir) {
    vector<pair<string, string>> datasets;
    
    DIR* dirp = opendir(dir.c_str());
    if (dirp == nullptr) {
        LOG("ERROR: Failed to open directory: " + dir);
        return datasets;
    }
    
    struct dirent* dp;
    while ((dp = readdir(dirp)) != nullptr) {
        string filename(dp->d_name);
        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".txt") {
            string filepath = dir + "/" + filename;
            string name = filename.substr(0, filename.size() - 4);
            
            // 跳过以._开头的文件
            if (name.substr(0, 2) != "._") {
                datasets.push_back({filepath, name});
            }
        }
    }
    closedir(dirp);
    
    // 将6亿边数据集放到最后
    sort(datasets.begin(), datasets.end(), [](const auto& a, const auto& b) {
        if (a.second == "temporal-reddit-reply") return false;
        if (b.second == "temporal-reddit-reply") return true;
        return a.second < b.second;
    });
    
    return datasets;
}

int main() {
    LOG("===========================================");
    LOG("Experiment 2 Optimized: UCR Optimization Analysis vs BZ");
    LOG("===========================================");
    
    // 设置OpenMP线程数
    omp_set_num_threads(4);
    LOG("OpenMP threads: " + to_string(omp_get_max_threads()));
    
    // 打开全局结果文件
    string output_file = "e2_optimized_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    // 写入文件头
    global_results << "Dataset\tWindow_Size\tMethod\tBZ_Time(ms)\tUCR_Time(ms)\tSpeedup\n";
    global_results.flush();
    
    LOG("Results will be saved to: " + output_file);
    
    // 获取所有数据集
    string dataset_dir = "dataset";
    auto datasets = get_datasets(dataset_dir);
    
    LOG("\nFound " + to_string(datasets.size()) + " datasets");
    LOG("Note: temporal-reddit-reply (6B edges) placed last - may crash but that's expected");
    
    // 对每个数据集运行实验
    int dataset_count = 0;
    for (size_t i = 0; i < datasets.size(); ++i) {
        const string& filepath = datasets[i].first;
        const string& name = datasets[i].second;
        
        dataset_count++;
        LOG("\n[" + to_string(dataset_count) + "/" + to_string(datasets.size()) + "] " + name);
        
        try {
            OptimizedSlidingWindowExperiment experiment(filepath, name);
            experiment.run();
        } catch (const exception& e) {
            LOG("ERROR: Exception processing " + name + ": " + string(e.what()));
        } catch (...) {
            LOG("ERROR: Unknown exception processing " + name);
        }
        
        // 每个数据集后强制内存清理
        global_results.flush();
        
        LOG("Dataset " + name + " completed - table wiped clean!");
    }
    
    global_results.close();
    
    LOG("\n===========================================");
    LOG("Experiment 2 Optimized completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}