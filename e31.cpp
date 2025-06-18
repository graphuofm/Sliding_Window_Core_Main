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

// 性能统计结构
struct PerformanceStats {
    double time_ms;
    size_t visited_vertices;
    size_t affected_vertices;
    size_t traversed_edges;
};

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
    
    // BZ算法计算核心度
    void compute_core_numbers_bz() {
        if (num_vertices == 0) return;
        
        fill(core.begin(), core.end(), 0);
        
        vector<int> degree(num_vertices);
        int max_degree = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
            degree[v] = adj[v].size();
            max_degree = max(max_degree, degree[v]);
        }
        
        if (max_degree == 0) return;
        
        vector<vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < num_vertices; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        vector<bool> processed(num_vertices, false);
        
        for (int d = 0; d <= max_degree; ++d) {
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
};

// MCD/PCD算法实现（带访问统计）
class MCDPCDCore {
private:
    Graph& G;
    vector<int> mcd_values;
    vector<int> pcd_values;
    
    // 统计变量
    unordered_set<vertex_t> visited_vertices;
    unordered_set<vertex_t> affected_vertices;
    size_t traversed_edges;
    
    bool is_qualified(vertex_t x, vertex_t y) const {
        int k = G.core[x];
        return G.core[y] > k || (G.core[y] == k && mcd_values[y] > k);
    }
    
    void track_vertex_visit(vertex_t v) {
        visited_vertices.insert(v);
        traversed_edges += G.adj[v].size();
    }
    
public:
    MCDPCDCore(Graph& g) : G(g), traversed_edges(0) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        mcd_values.clear();
        pcd_values.clear();
        mcd_values.resize(n, 0);
        pcd_values.resize(n, 0);
        
        // 计算MCD值
        for (vertex_t v = 0; v < n; ++v) {
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] >= G.core[v]) {
                    ++count;
                }
            }
            mcd_values[v] = count;
        }
        
        // 计算PCD值
        for (vertex_t v = 0; v < n; ++v) {
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (is_qualified(v, w)) {
                    ++count;
                }
            }
            pcd_values[v] = count;
        }
    }
    
    void clear_stats() {
        visited_vertices.clear();
        affected_vertices.clear();
        traversed_edges = 0;
    }
    
    PerformanceStats get_stats(double time_ms) const {
        return {time_ms, visited_vertices.size(), affected_vertices.size(), traversed_edges};
    }
    
    // 批量处理边的插入
    PerformanceStats process_batch_insert(const vector<edge_t>& edges) {
        clear_stats();
        Timer timer;
        
        // 添加边到图中
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            G.add_edge(u, v);
            
            // 更新MCD值
            if (G.core[v] >= G.core[u]) mcd_values[u]++;
            if (G.core[u] >= G.core[v]) mcd_values[v]++;
            
            track_vertex_visit(u);
            track_vertex_visit(v);
        }
        
        // 重新计算PCD值
        reset();
        
        // 找候选顶点
        unordered_set<vertex_t> candidates;
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (G.core[u] > G.core[v]) {
                if (pcd_values[v] > G.core[v]) {
                    candidates.insert(v);
                }
            } else if (G.core[u] < G.core[v]) {
                if (pcd_values[u] > G.core[u]) {
                    candidates.insert(u);
                }
            } else {
                if (pcd_values[u] > G.core[u]) {
                    candidates.insert(u);
                }
                if (pcd_values[v] > G.core[v]) {
                    candidates.insert(v);
                }
            }
        }
        
        // 处理候选顶点升级
        for (vertex_t v : candidates) {
            process_vertex_insertion(v);
        }
        
        // 最终重置
        reset();
        
        double time_ms = timer.elapsed_milliseconds();
        return get_stats(time_ms);
    }
    
    // 批量处理边的删除
    PerformanceStats process_batch_remove(const vector<edge_t>& edges) {
        clear_stats();
        Timer timer;
        
        // 收集受影响的顶点
        unordered_map<int, unordered_set<vertex_t>> k_groups;
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                // 更新MCD值
                if (G.core[v] >= G.core[u]) mcd_values[u]--;
                if (G.core[u] >= G.core[v]) mcd_values[v]--;
                
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku > 0) k_groups[ku].insert(u);
                if (kv > 0) k_groups[kv].insert(v);
                
                G.remove_edge(u, v);
                
                track_vertex_visit(u);
                track_vertex_visit(v);
            }
        }
        
        // 按核心度层次处理删除
        vector<int> k_values;
        for (const auto& pair : k_groups) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            process_core_level_removal(k, k_groups[k]);
        }
        
        // 最终重置
        reset();
        
        double time_ms = timer.elapsed_milliseconds();
        return get_stats(time_ms);
    }
    
private:
    void process_vertex_insertion(vertex_t root) {
        int k = G.core[root];
        
        if (pcd_values[root] <= k) {
            return;
        }
        
        // 找连通分量
        unordered_set<vertex_t> component;
        queue<vertex_t> q;
        unordered_set<vertex_t> visited;
        
        q.push(root);
        visited.insert(root);
        track_vertex_visit(root);
        
        while (!q.empty()) {
            vertex_t v = q.front();
            q.pop();
            
            if (G.core[v] == k) {
                component.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    track_vertex_visit(w);
                    if (G.core[w] == k && !visited.count(w)) {
                        visited.insert(w);
                        q.push(w);
                    }
                }
            }
        }
        
        // 模拟升级过程
        unordered_map<vertex_t, int> cd;
        for (vertex_t v : component) {
            cd[v] = pcd_values[v];
        }
        
        unordered_set<vertex_t> evicted;
        bool stable = false;
        
        while (!stable) {
            stable = true;
            vector<vertex_t> to_evict;
            
            for (vertex_t v : component) {
                if (evicted.count(v)) continue;
                if (cd[v] <= k) {
                    to_evict.push_back(v);
                    stable = false;
                }
            }
            
            for (vertex_t v : to_evict) {
                evicted.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (component.count(w) && !evicted.count(w) && 
                        G.core[w] == k && is_qualified(w, v)) {
                        cd[w]--;
                    }
                }
            }
        }
        
        // 升级剩余顶点
        for (vertex_t v : component) {
            if (!evicted.count(v)) {
                G.core[v]++;
                affected_vertices.insert(v);
            }
        }
    }
    
    void process_core_level_removal(int k, const unordered_set<vertex_t>& seeds) {
        unordered_set<vertex_t> visited;
        
        for (vertex_t seed : seeds) {
            if (visited.count(seed) || G.core[seed] != k) continue;
            
            // 找连通分量
            vector<vertex_t> component;
            unordered_map<vertex_t, int> eff_deg;
            queue<vertex_t> q;
            
            q.push(seed);
            visited.insert(seed);
            track_vertex_visit(seed);
            
            while (!q.empty()) {
                vertex_t v = q.front();
                q.pop();
                
                if (G.core[v] == k) {
                    component.push_back(v);
                    
                    int count = 0;
                    for (vertex_t w : G.adj[v]) {
                        track_vertex_visit(w);
                        if (G.core[w] >= k) count++;
                    }
                    eff_deg[v] = count;
                    
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] == k && !visited.count(w)) {
                            visited.insert(w);
                            q.push(w);
                        }
                    }
                }
            }
            
            // 级联降级
            queue<vertex_t> degrade_queue;
            unordered_set<vertex_t> degraded;
            
            for (vertex_t v : component) {
                if (eff_deg[v] < k) {
                    degrade_queue.push(v);
                }
            }
            
            while (!degrade_queue.empty()) {
                vertex_t v = degrade_queue.front();
                degrade_queue.pop();
                
                if (G.core[v] != k || degraded.count(v)) continue;
                
                G.core[v] = k - 1;
                degraded.insert(v);
                affected_vertices.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k && !degraded.count(w)) {
                        eff_deg[w]--;
                        if (eff_deg[w] < k) {
                            degrade_queue.push(w);
                        }
                    }
                }
            }
        }
    }
};

// UCR算法实现（带访问统计）
class UCRCore {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    
    // 统计变量
    unordered_set<vertex_t> visited_vertices;
    unordered_set<vertex_t> affected_vertices;
    size_t traversed_edges;
    
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
    
    void track_vertex_visit(vertex_t v) {
        visited_vertices.insert(v);
        traversed_edges += G.adj[v].size();
    }
    
public:
    UCRCore(Graph& g) : G(g), traversed_edges(0) {
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
    
    void clear_stats() {
        visited_vertices.clear();
        affected_vertices.clear();
        traversed_edges = 0;
    }
    
    PerformanceStats get_stats(double time_ms) const {
        return {time_ms, visited_vertices.size(), affected_vertices.size(), traversed_edges};
    }
    
    // 批量处理边的插入
    PerformanceStats process_batch_insert(const vector<edge_t>& edges) {
        clear_stats();
        Timer timer;
        
        // 扩展数据结构
        vertex_t max_vertex_id = 0;
        for (const auto& edge : edges) {
            max_vertex_id = max(max_vertex_id, max(edge.first, edge.second));
        }
        
        if (max_vertex_id >= G.core.size()) {
            G.core.resize(max_vertex_id + 1, 0);
        }
        if (max_vertex_id >= r_values.size()) {
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // 添加边并更新r值
        unordered_set<vertex_t> immediate_affected;
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.ensure_vertex(max(u, v));
            
            // 为新顶点设置初始核心度
            if (G.adj[u].empty()) G.core[u] = 1;
            if (G.adj[v].empty()) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
            
            immediate_affected.insert(u);
            immediate_affected.insert(v);
            track_vertex_visit(u);
            track_vertex_visit(v);
        }
        
        // 批量更新s值
        unordered_set<vertex_t> s_update_set;
        for (vertex_t v : immediate_affected) {
            s_update_set.insert(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    s_update_set.insert(w);
                    track_vertex_visit(w);
                }
            }
        }
        
        for (vertex_t v : s_update_set) {
            update_s_value(v);
        }
        
        // 找候选顶点
        unordered_set<vertex_t> candidates;
        for (vertex_t v : immediate_affected) {
            if (v < r_values.size() && is_qualified(v)) {
                candidates.insert(v);
            }
        }
        
        // 批量处理升级
        if (!candidates.empty()) {
            vector<vector<vertex_t>> components;
            identify_components(candidates, components);
            
            for (const auto& component : components) {
                process_component_upgrade(component);
            }
        }
        
        double time_ms = timer.elapsed_milliseconds();
        return get_stats(time_ms);
    }
    
    // 批量处理边的删除
    PerformanceStats process_batch_remove(const vector<edge_t>& edges) {
        clear_stats();
        Timer timer;
        
        // 更新r值并删除边
        unordered_map<int, unordered_set<vertex_t>> k_groups;
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                if (ku > 0) k_groups[ku].insert(u);
                if (kv > 0) k_groups[kv].insert(v);
                
                G.remove_edge(u, v);
                
                track_vertex_visit(u);
                track_vertex_visit(v);
            }
        }
        
        // 按核心度层次处理删除
        vector<int> k_values;
        for (const auto& pair : k_groups) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            process_core_level_removal(k, k_groups[k]);
        }
        
        double time_ms = timer.elapsed_milliseconds();
        return get_stats(time_ms);
    }
    
private:
    void identify_components(const unordered_set<vertex_t>& candidates,
                            vector<vector<vertex_t>>& components) {
        unordered_set<vertex_t> visited;
        
        for (vertex_t v : candidates) {
            if (visited.count(v)) continue;
            
            vector<vertex_t> component;
            queue<vertex_t> q;
            q.push(v);
            visited.insert(v);
            
            int k = G.core[v];
            
            while (!q.empty()) {
                vertex_t u = q.front();
                q.pop();
                
                if (G.core[u] == k) {
                    component.push_back(u);
                    track_vertex_visit(u);
                    
                    for (vertex_t w : G.adj[u]) {
                        if (G.core[w] == k && !visited.count(w) && candidates.count(w)) {
                            visited.insert(w);
                            q.push(w);
                        }
                    }
                }
            }
            
            if (!component.empty()) {
                components.push_back(move(component));
            }
        }
    }
    
    void process_component_upgrade(const vector<vertex_t>& component) {
        if (component.empty()) return;
        
        int k = G.core[component[0]];
        
        unordered_map<vertex_t, int> cd;
        for (vertex_t v : component) {
            if (v < r_values.size()) {
                cd[v] = r_values[v] + s_values[v];
            }
        }
        
        unordered_set<vertex_t> evicted;
        bool stable = false;
        
        while (!stable) {
            stable = true;
            vector<vertex_t> to_evict;
            
            for (vertex_t v : component) {
                if (evicted.count(v)) continue;
                if (cd[v] <= k) {
                    to_evict.push_back(v);
                    stable = false;
                }
            }
            
            for (vertex_t v : to_evict) {
                evicted.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k && !evicted.count(w) && cd.count(w)) {
                        cd[w]--;
                    }
                }
            }
        }
        
        // 批量升级
        for (vertex_t v : component) {
            if (!evicted.count(v)) {
                G.core[v]++;
                affected_vertices.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (w < r_values.size() && G.core[w] < G.core[v]) {
                        r_values[w]++;
                    }
                }
            }
        }
    }
    
    void process_core_level_removal(int k, const unordered_set<vertex_t>& seeds) {
        unordered_set<vertex_t> visited;
        
        for (vertex_t seed : seeds) {
            if (visited.count(seed) || G.core[seed] != k) continue;
            
            // 找连通分量
            vector<vertex_t> component;
            unordered_map<vertex_t, int> eff_deg;
            queue<vertex_t> q;
            
            q.push(seed);
            visited.insert(seed);
            track_vertex_visit(seed);
            
            while (!q.empty()) {
                vertex_t v = q.front();
                q.pop();
                
                if (G.core[v] == k) {
                    component.push_back(v);
                    
                    int count = 0;
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] >= k) count++;
                    }
                    eff_deg[v] = count;
                    
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] == k && !visited.count(w)) {
                            visited.insert(w);
                            q.push(w);
                            track_vertex_visit(w);
                        }
                    }
                }
            }
            
            // 批量级联降级
            queue<vertex_t> degrade_queue;
            
            for (vertex_t v : component) {
                if (eff_deg[v] < k) {
                    degrade_queue.push(v);
                }
            }
            
            while (!degrade_queue.empty()) {
                vertex_t v = degrade_queue.front();
                degrade_queue.pop();
                
                if (G.core[v] != k) continue;
                
                G.core[v] = k - 1;
                affected_vertices.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k) {
                        eff_deg[w]--;
                        if (eff_deg[w] < k) {
                            degrade_queue.push(w);
                        }
                    }
                    
                    if (w < r_values.size() && G.core[w] > k - 1) {
                        r_values[w]--;
                    }
                }
            }
        }
    }
};

// 批量性能和空间对比实验类
class BatchSpaceExperiment {
private:
    string dataset_path;
    string dataset_name;
    
public:
    BatchSpaceExperiment(const string& path, const string& name) 
        : dataset_path(path), dataset_name(name) {}
    
    void run() {
        LOG("=====================================");
        LOG("Dataset: " + dataset_name);
        LOG("=====================================");
        
        // 加载前100,000条边构建基础图
        auto edges = load_initial_edges(100000);
        if (edges.empty()) {
            LOG("ERROR: Failed to load initial edges for " + dataset_name);
            return;
        }
        
        LOG("Loaded " + to_string(edges.size()) + " edges for initial graph");
        
        // 构建初始图
        Graph initial_graph;
        for (const auto& e : edges) {
            initial_graph.add_edge(e.src, e.dst);
        }
        initial_graph.compute_core_numbers_bz();
        
        LOG("Initial graph: " + to_string(initial_graph.num_vertices) + " vertices");
        
        // 测试不同批量大小，扩展到10000
        vector<int> batch_sizes = {1, 10, 100, 1000, 5000, 10000};
        
        for (int batch_size : batch_sizes) {
            run_batch_experiment(initial_graph, batch_size);
        }
    }
    
private:
    vector<TemporalEdge> load_initial_edges(size_t limit) {
        vector<TemporalEdge> edges;
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            LOG("ERROR: Cannot open file: " + dataset_path);
            return edges;
        }
        
        string line;
        size_t loaded = 0;
        
        while (getline(file, line) && loaded < limit) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                edges.emplace_back(src, dst, ts);
                loaded++;
            }
        }
        
        file.close();
        
        // 按时间戳排序
        sort(edges.begin(), edges.end(), 
             [](const TemporalEdge& a, const TemporalEdge& b) {
                 return a.timestamp < b.timestamp;
             });
        
        return edges;
    }
    
    void run_batch_experiment(const Graph& base_graph, int batch_size) {
        LOG("\n--- Batch size: " + to_string(batch_size) + " edges ---");
        
        // 生成测试边
        auto test_edges = generate_test_edges(base_graph, batch_size);
        
        if (test_edges.first.empty() && test_edges.second.empty()) {
            LOG("WARNING: No test edges generated for batch size " + to_string(batch_size));
            return;
        }
        
        // MCD/PCD测试
        Graph mcd_graph = base_graph.copy();
        MCDPCDCore mcd_core(mcd_graph);
        
        auto mcd_insert_stats = mcd_core.process_batch_insert(test_edges.first);
        auto mcd_remove_stats = mcd_core.process_batch_remove(test_edges.second);
        
        LOG("  MCD/PCD Insert:");
        LOG("    Time: " + to_string(mcd_insert_stats.time_ms) + " ms");
        LOG("    Visited vertices: " + to_string(mcd_insert_stats.visited_vertices));
        LOG("    Affected vertices: " + to_string(mcd_insert_stats.affected_vertices));
        LOG("    Traversed edges: " + to_string(mcd_insert_stats.traversed_edges));
        
        LOG("  MCD/PCD Remove:");
        LOG("    Time: " + to_string(mcd_remove_stats.time_ms) + " ms");
        LOG("    Visited vertices: " + to_string(mcd_remove_stats.visited_vertices));
        LOG("    Affected vertices: " + to_string(mcd_remove_stats.affected_vertices));
        LOG("    Traversed edges: " + to_string(mcd_remove_stats.traversed_edges));
        
        // UCR测试
        Graph ucr_graph = base_graph.copy();
        UCRCore ucr_core(ucr_graph);
        
        auto ucr_insert_stats = ucr_core.process_batch_insert(test_edges.first);
        auto ucr_remove_stats = ucr_core.process_batch_remove(test_edges.second);
        
        LOG("  UCR Insert:");
        LOG("    Time: " + to_string(ucr_insert_stats.time_ms) + " ms");
        LOG("    Visited vertices: " + to_string(ucr_insert_stats.visited_vertices));
        LOG("    Affected vertices: " + to_string(ucr_insert_stats.affected_vertices));
        LOG("    Traversed edges: " + to_string(ucr_insert_stats.traversed_edges));
        
        LOG("  UCR Remove:");
        LOG("    Time: " + to_string(ucr_remove_stats.time_ms) + " ms");
        LOG("    Visited vertices: " + to_string(ucr_remove_stats.visited_vertices));
        LOG("    Affected vertices: " + to_string(ucr_remove_stats.affected_vertices));
        LOG("    Traversed edges: " + to_string(ucr_remove_stats.traversed_edges));
        
        // 计算比较指标
        double insert_time_speedup = (mcd_insert_stats.time_ms > 0) ? 
            mcd_insert_stats.time_ms / ucr_insert_stats.time_ms : 1.0;
        double remove_time_speedup = (mcd_remove_stats.time_ms > 0) ? 
            mcd_remove_stats.time_ms / ucr_remove_stats.time_ms : 1.0;
        
        double insert_space_reduction = (mcd_insert_stats.visited_vertices > 0) ?
            (double)ucr_insert_stats.visited_vertices / mcd_insert_stats.visited_vertices : 1.0;
        double remove_space_reduction = (mcd_remove_stats.visited_vertices > 0) ?
            (double)ucr_remove_stats.visited_vertices / mcd_remove_stats.visited_vertices : 1.0;
        
        LOG("  Comparison:");
        LOG("    Insert time speedup: " + to_string(insert_time_speedup) + "x");
        LOG("    Remove time speedup: " + to_string(remove_time_speedup) + "x");
        LOG("    Insert space reduction: " + to_string(insert_space_reduction * 100) + "%");
        LOG("    Remove space reduction: " + to_string(remove_space_reduction * 100) + "%");
        
        // 写入全局结果文件
        global_results << dataset_name << "\t" 
                      << batch_size << "\t"
                      << fixed << setprecision(2)
                      << mcd_insert_stats.time_ms << "\t"
                      << ucr_insert_stats.time_ms << "\t"
                      << insert_time_speedup << "\t"
                      << mcd_insert_stats.visited_vertices << "\t"
                      << ucr_insert_stats.visited_vertices << "\t"
                      << insert_space_reduction << "\t"
                      << mcd_remove_stats.time_ms << "\t"
                      << ucr_remove_stats.time_ms << "\t"
                      << remove_time_speedup << "\t"
                      << mcd_remove_stats.visited_vertices << "\t"
                      << ucr_remove_stats.visited_vertices << "\t"
                      << remove_space_reduction << "\n";
        global_results.flush();
        
        // 清理图
        mcd_graph.clear();
        ucr_graph.clear();
    }
    
    pair<vector<edge_t>, vector<edge_t>> generate_test_edges(const Graph& graph, int batch_size) {
        vector<edge_t> insert_edges, remove_edges;
        
        // 生成插入边：随机选择不存在的边
        set<pair<vertex_t, vertex_t>> existing_edges;
        for (vertex_t u = 0; u < graph.num_vertices; ++u) {
            for (vertex_t v : graph.adj[u]) {
                if (u < v) {
                    existing_edges.insert({u, v});
                }
            }
        }
        
        // 随机生成新边
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<vertex_t> vertex_dist(0, graph.num_vertices - 1);
        
        int attempts = 0;
        while (insert_edges.size() < batch_size && attempts < batch_size * 100) {
            vertex_t u = vertex_dist(gen);
            vertex_t v = vertex_dist(gen);
            attempts++;
            
            if (u != v && u < v && existing_edges.find({u, v}) == existing_edges.end()) {
                insert_edges.push_back({u, v});
                existing_edges.insert({u, v});
            }
        }
        
        // 生成删除边：从现有边中随机选择
        vector<pair<vertex_t, vertex_t>> all_edges;
        for (vertex_t u = 0; u < graph.num_vertices; ++u) {
            for (vertex_t v : graph.adj[u]) {
                if (u < v) {
                    all_edges.push_back({u, v});
                }
            }
        }
        
        if (!all_edges.empty()) {
            shuffle(all_edges.begin(), all_edges.end(), gen);
            int remove_count = min(batch_size, static_cast<int>(all_edges.size()));
            remove_edges.assign(all_edges.begin(), all_edges.begin() + remove_count);
        }
        
        return {insert_edges, remove_edges};
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
    
    // 按文件名排序
    sort(datasets.begin(), datasets.end());
    
    return datasets;
}

int main() {
    LOG("===========================================");
    LOG("Experiment 3.1: UCR vs MCD/PCD Performance and Space Analysis");
    LOG("Extended batch sizes up to 10,000 edges");
    LOG("===========================================");
    
    // 打开全局结果文件
    string output_file = "e31_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    // 写入文件头
    global_results << "Dataset\tBatch_Size\t"
                   << "MCD_Insert_Time(ms)\tUCR_Insert_Time(ms)\tInsert_Time_Speedup\t"
                   << "MCD_Insert_Visited\tUCR_Insert_Visited\tInsert_Space_Reduction\t"
                   << "MCD_Remove_Time(ms)\tUCR_Remove_Time(ms)\tRemove_Time_Speedup\t"
                   << "MCD_Remove_Visited\tUCR_Remove_Visited\tRemove_Space_Reduction\n";
    global_results.flush();
    
    LOG("Results will be saved to: " + output_file);
    
    // 获取所有数据集
    string dataset_dir = "dataset";
    auto datasets = get_datasets(dataset_dir);
    
    LOG("\nFound " + to_string(datasets.size()) + " datasets");
    
    // 对每个数据集运行实验
    int dataset_count = 0;
    for (size_t i = 0; i < datasets.size(); ++i) {
        const string& filepath = datasets[i].first;
        const string& name = datasets[i].second;
        
        dataset_count++;
        LOG("\n[" + to_string(dataset_count) + "/" + to_string(datasets.size()) + "] " + name);
        
        try {
            BatchSpaceExperiment experiment(filepath, name);
            experiment.run();
        } catch (const exception& e) {
            LOG("ERROR: Exception processing " + name + ": " + string(e.what()));
        } catch (...) {
            LOG("ERROR: Unknown exception processing " + name);
        }
        
        global_results.flush();
    }
    
    global_results.close();
    
    LOG("\n===========================================");
    LOG("Experiment 3.1 completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}