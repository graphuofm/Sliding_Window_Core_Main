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
#include <dirent.h>
#include <sys/stat.h>
#include <cmath>

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
        ensure_vertex(max(u, v));
        
        // 避免自环和重复边
        if (u == v) return;
        
        if (find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
            adj[u].push_back(v);
        }
        if (find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) {
            adj[v].push_back(u);
        }
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        if (u >= num_vertices || v >= num_vertices) return;
        
        // 安全的删除方式
        auto& adj_u = adj[u];
        auto it_u = find(adj_u.begin(), adj_u.end(), v);
        if (it_u != adj_u.end()) {
            adj_u.erase(it_u);
        }
        
        auto& adj_v = adj[v];
        auto it_v = find(adj_v.begin(), adj_v.end(), u);
        if (it_v != adj_v.end()) {
            adj_v.erase(it_v);
        }
    }
    
    size_t degree(vertex_t v) const {
        if (v >= num_vertices) return 0;
        return adj[v].size();
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
    
    size_t num_edges() const {
        size_t edges = 0;
        for (const auto& neighbors : adj) {
            edges += neighbors.size();
        }
        return edges / 2;
    }
};

// MCD/PCD算法实现
class MCDPCDCore {
private:
    Graph& G;
    vector<int> MCD;
    vector<int> PCD;
    
    bool qual(vertex_t v, vertex_t w) const {
        if (v >= G.num_vertices || w >= G.num_vertices) return false;
        int k = G.core[v];
        return G.core[w] > k || (G.core[w] == k && MCD[w] > k);
    }
    
    void recompute_MCD(const unordered_set<vertex_t>& vertices) {
        for (vertex_t v : vertices) {
            if (v >= G.num_vertices) continue;
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] >= G.core[v]) {
                    ++count;
                }
            }
            MCD[v] = count;
        }
    }
    
    void recompute_PCD(const unordered_set<vertex_t>& vertices) {
        for (vertex_t v : vertices) {
            if (v >= G.num_vertices) continue;
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
    MCDPCDCore(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        MCD.resize(n, 0);
        PCD.resize(n, 0);
        
        // 初始化MCD和PCD
        for (vertex_t v = 0; v < n; ++v) {
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] >= G.core[v]) {
                    ++count;
                }
            }
            MCD[v] = count;
        }
        
        for (vertex_t v = 0; v < n; ++v) {
            int count = 0;
            for (vertex_t w : G.adj[v]) {
                if (qual(v, w)) {
                    ++count;
                }
            }
            PCD[v] = count;
        }
    }
    
    double process_batch(const vector<edge_t>& remove_edges, 
                        const vector<edge_t>& add_edges) {
        Timer timer;
        Timer total_timer;  // 用于检测超时
        const double TIMEOUT_MS = 300000.0;  // 5分钟 = 300秒 = 300000毫秒
        
        // 处理删除
        unordered_set<vertex_t> affected_by_removal;
        for (const auto& edge : remove_edges) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u >= G.num_vertices || v >= G.num_vertices) continue;
            
            affected_by_removal.insert(u);
            affected_by_removal.insert(v);
            
            // 更新MCD
            if (G.core[v] >= G.core[u]) --MCD[u];
            if (G.core[u] >= G.core[v]) --MCD[v];
            
            G.remove_edge(u, v);
        }
        
        // 重新计算受影响顶点的PCD
        unordered_set<vertex_t> to_update = affected_by_removal;
        for (vertex_t v : affected_by_removal) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            for (vertex_t w : G.adj[v]) {
                to_update.insert(w);
            }
        }
        recompute_PCD(to_update);
        
        // 处理核心度降级
        queue<vertex_t> degrade_queue;
        unordered_set<vertex_t> degraded;
        
        for (vertex_t v : affected_by_removal) {
            if (v < G.num_vertices && MCD[v] < G.core[v]) {
                degrade_queue.push(v);
            }
        }
        
        while (!degrade_queue.empty()) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            
            vertex_t v = degrade_queue.front();
            degrade_queue.pop();
            
            if (degraded.count(v)) continue;
            
            int old_core = G.core[v];
            G.core[v]--;
            degraded.insert(v);
            
            // 更新邻居
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == old_core) {
                    MCD[w]--;
                    if (MCD[w] < old_core && !degraded.count(w)) {
                        degrade_queue.push(w);
                    }
                }
            }
        }
        
        // 处理插入
        vertex_t max_vertex = 0;
        for (const auto& edge : add_edges) {
            max_vertex = max(max_vertex, max(edge.first, edge.second));
        }
        
        if (max_vertex >= G.num_vertices) {
            G.ensure_vertex(max_vertex);
            MCD.resize(G.num_vertices, 0);
            PCD.resize(G.num_vertices, 0);
        }
        
        unordered_set<vertex_t> affected_by_insertion;
        for (const auto& edge : add_edges) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.add_edge(u, v);
            
            affected_by_insertion.insert(u);
            affected_by_insertion.insert(v);
            
            // 更新MCD
            if (G.core[v] >= G.core[u]) ++MCD[u];
            if (G.core[u] >= G.core[v]) ++MCD[v];
        }
        
        // 重新计算PCD
        to_update = affected_by_insertion;
        for (vertex_t v : affected_by_insertion) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            for (vertex_t w : G.adj[v]) {
                to_update.insert(w);
            }
        }
        recompute_PCD(to_update);
        
        // 处理核心度升级
        unordered_set<vertex_t> candidates;
        for (vertex_t v : affected_by_insertion) {
            if (v < G.num_vertices && PCD[v] > G.core[v]) {
                candidates.insert(v);
            }
        }
        
        // 找连通分量并处理
        unordered_set<vertex_t> visited;
        for (vertex_t start : candidates) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            if (visited.count(start)) continue;
            
            // BFS找连通分量
            queue<vertex_t> q;
            vector<vertex_t> component;
            int k = G.core[start];
            
            q.push(start);
            visited.insert(start);
            
            while (!q.empty()) {
                if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
                
                vertex_t v = q.front();
                q.pop();
                
                if (G.core[v] == k && candidates.count(v)) {
                    component.push_back(v);
                    
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] == k && !visited.count(w)) {
                            visited.insert(w);
                            q.push(w);
                        }
                    }
                }
            }
            
            // 模拟升级
            unordered_map<vertex_t, int> cd;
            for (vertex_t v : component) {
                cd[v] = PCD[v];
            }
            
            unordered_set<vertex_t> evicted;
            bool stable = false;
            
            while (!stable) {
                if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
                stable = true;
                
                for (vertex_t v : component) {
                    if (evicted.count(v)) continue;
                    if (cd[v] <= k) {
                        evicted.insert(v);
                        stable = false;
                        
                        for (vertex_t w : G.adj[v]) {
                            if (G.core[w] == k && cd.count(w) && !evicted.count(w)) {
                                cd[w]--;
                            }
                        }
                    }
                }
            }
            
            // 升级
            for (vertex_t v : component) {
                if (!evicted.count(v)) {
                    G.core[v]++;
                }
            }
        }
        
        // 最终更新MCD/PCD
        unordered_set<vertex_t> final_update;
        final_update.insert(affected_by_removal.begin(), affected_by_removal.end());
        final_update.insert(affected_by_insertion.begin(), affected_by_insertion.end());
        
        for (vertex_t v : final_update) {
            if (total_timer.elapsed_milliseconds() > TIMEOUT_MS) return -1;
            for (vertex_t w : G.adj[v]) {
                final_update.insert(w);
            }
        }
        
        recompute_MCD(final_update);
        recompute_PCD(final_update);
        
        return timer.elapsed_milliseconds();
    }
};

// UCR算法实现（批处理优化版本）
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
        
        s_values[v] = s_count;
    }
    
public:
    UCRBatch(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
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
    
    // 批量处理边的删除和插入
    double process_batch(const vector<edge_t>& remove_edges, 
                        const vector<edge_t>& add_edges) {
        Timer timer;
        
        // Phase 1: 批量删除边
        unordered_map<int, unordered_set<vertex_t>> k_groups_remove;
        
        // 更新r值并删除边
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv) r_values[u]--;
                else if (ku > kv) r_values[v]--;
                
                if (ku > 0) k_groups_remove[ku].insert(u);
                if (kv > 0) k_groups_remove[kv].insert(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // 按核心度层次处理删除
        vector<int> k_values;
        for (const auto& pair : k_groups_remove) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            process_core_level_removal(k, k_groups_remove[k]);
        }
        
        // Phase 2: 批量插入边
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
        
        // 添加边并收集受影响顶点
        unordered_set<vertex_t> affected_vertices;
        
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            G.ensure_vertex(max(u, v));
            
            if (u == v) continue;  // 跳过自环
            
            if (G.adj[u].size() == 0) G.core[u] = 1;
            if (G.adj[v].size() == 0) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv) r_values[u]++;
            else if (ku > kv) r_values[v]++;
            
            affected_vertices.insert(u);
            affected_vertices.insert(v);
        }
        
        // 批量更新s值
        unordered_set<vertex_t> s_update_set;
        for (vertex_t v : affected_vertices) {
            s_update_set.insert(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    s_update_set.insert(w);
                }
            }
        }
        
        for (vertex_t v : s_update_set) {
            update_s_value(v);
        }
        
        // 找所有可能升级的候选
        unordered_set<vertex_t> all_candidates;
        for (vertex_t v : affected_vertices) {
            if (is_qualified(v)) {
                all_candidates.insert(v);
            }
        }
        
        // 批量处理升级
        if (!all_candidates.empty()) {
            vector<vector<vertex_t>> components;
            identify_components(all_candidates, components);
            
            for (const auto& component : components) {
                process_component_upgrade(component);
            }
        }
        
        return timer.elapsed_milliseconds();
    }
    
private:
    void process_core_level_removal(int k, const unordered_set<vertex_t>& seeds) {
        unordered_set<vertex_t> visited;
        
        for (vertex_t seed : seeds) {
            if (visited.count(seed) || G.core[seed] != k) continue;
            
            // 找连通分量
            queue<vertex_t> q;
            q.push(seed);
            visited.insert(seed);
            
            vector<vertex_t> component;
            unordered_map<vertex_t, int> eff_deg;
            
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
                        }
                    }
                }
            }
            
            // 批量级联降级
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
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k && !degraded.count(w)) {
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
            
            // 批量更新s值
            unordered_set<vertex_t> update_set;
            for (vertex_t v : degraded) {
                update_set.insert(v);
                for (vertex_t w : G.adj[v]) {
                    update_set.insert(w);
                }
            }
            
            for (vertex_t v : update_set) {
                update_s_value(v);
            }
        }
    }
    
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
            cd[v] = r_values[v] + s_values[v];
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
                
                for (vertex_t w : G.adj[v]) {
                    if (w < r_values.size() && G.core[w] < G.core[v]) {
                        r_values[w]++;
                    }
                }
            }
        }
        
        // 批量更新s值
        unordered_set<vertex_t> update_set;
        for (vertex_t v : component) {
            if (!evicted.count(v)) {
                update_set.insert(v);
                for (vertex_t w : G.adj[v]) {
                    update_set.insert(w);
                }
            }
        }
        
        for (vertex_t v : update_set) {
            update_s_value(v);
        }
    }
};

// 滑动窗口批量大小实验类
class SlidingWindowBatchExperiment {
private:
    string dataset_path;
    string dataset_name;
    vector<TemporalEdge> edges;
    vector<vector<edge_t>> bins;
    
    static constexpr int WINDOW_SIZE = 50;       // 固定窗口大小为50 bins
    static constexpr int NUM_BINS = 1000;        // 总共1000个bins
    
public:
    SlidingWindowBatchExperiment(const string& path, const string& name) 
        : dataset_path(path), dataset_name(name) {}
    
    void run() {
        LOG("=====================================");
        LOG("Dataset: " + dataset_name);
        LOG("=====================================");
        
        // 加载数据
        if (!load_dataset()) {
            LOG("ERROR: Failed to load dataset " + dataset_name);
            return;
        }
        
        // 创建bins
        create_bins(NUM_BINS);
        
        // 选择合适的起始位置
        int starting_bin = 400;  // 从中间位置开始
        if (starting_bin + WINDOW_SIZE > bins.size()) {
            starting_bin = bins.size() - WINDOW_SIZE - 100;
        }
        if (starting_bin < 0) starting_bin = 0;
        
        // 构建初始窗口图
        LOG("Building initial window (bins " + to_string(starting_bin) + 
            " to " + to_string(starting_bin + WINDOW_SIZE - 1) + ")...");
        Graph initial_graph;
        size_t total_edges = 0;
        
        for (int i = starting_bin; i < starting_bin + WINDOW_SIZE; ++i) {
            total_edges += bins[i].size();
            for (const auto& edge : bins[i]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
        }
        
        LOG("  Initial window edges: " + to_string(total_edges));
        LOG("  Initial vertices: " + to_string(initial_graph.num_vertices));
        
        // 计算初始核心度
        Timer init_timer;
        initial_graph.compute_core_numbers_bz();
        LOG("  Initial core computation: " + to_string(init_timer.elapsed_milliseconds()) + " ms");
        
        // 测试不同的批量大小
        vector<int> batch_sizes = {1, 10, 100};  // 修改：去掉1000
        
        for (int batch_size : batch_sizes) {
            run_batch_size_experiment(initial_graph, starting_bin, batch_size);
        }
    }
    
private:
    bool load_dataset() {
        Timer timer;
        LOG("Loading dataset from: " + dataset_path);
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return false;
        }
        
        string line;
        edges.clear();
        size_t line_count = 0;
        
        while (getline(file, line)) {
            line_count++;
            if (line_count % 1000000 == 0) {
                LOG("  Loaded " + to_string(line_count) + " lines...");
            }
            
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts) {
                edges.emplace_back(src, dst, ts);
            }
        }
        
        file.close();
        
        LOG("  Total edges loaded: " + to_string(edges.size()));
        LOG("  Load time: " + to_string(timer.elapsed_milliseconds()) + " ms");
        
        return !edges.empty();
    }
    
    void create_bins(int num_bins) {
        if (edges.empty()) {
            LOG("ERROR: No edges to bin");
            return;
        }
        
        LOG("Creating " + to_string(num_bins) + " bins...");
        
        // 按时间戳排序
        sort(edges.begin(), edges.end(), 
             [](const TemporalEdge& a, const TemporalEdge& b) {
                 return a.timestamp < b.timestamp;
             });
        
        timestamp_t min_ts = edges.front().timestamp;
        timestamp_t max_ts = edges.back().timestamp;
        
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / num_bins;
        bins.clear();
        bins.resize(num_bins);
        
        for (const auto& e : edges) {
            int bin_idx = min(static_cast<int>((e.timestamp - min_ts) / bin_span), num_bins - 1);
            bins[bin_idx].push_back({e.src, e.dst});
        }
        
        LOG("  Binning complete");
    }
    
    void run_batch_size_experiment(const Graph& initial_graph, int starting_bin, int batch_size) {
        LOG("\n--- Batch size: " + to_string(batch_size) + " edges ---");
        
        // 准备批次：收集边直到达到批量大小
        vector<edge_t> remove_batch, add_batch;
        
        // 从多个bins收集边，直到达到批量大小
        int current_bin = starting_bin;
        while (remove_batch.size() < batch_size && current_bin < bins.size()) {
            for (const auto& edge : bins[current_bin]) {
                remove_batch.push_back(edge);
                if (remove_batch.size() >= batch_size) break;
            }
            current_bin++;
        }
        
        // 收集要添加的边
        current_bin = starting_bin + WINDOW_SIZE;
        while (add_batch.size() < batch_size && current_bin < bins.size()) {
            for (const auto& edge : bins[current_bin]) {
                add_batch.push_back(edge);
                if (add_batch.size() >= batch_size) break;
            }
            current_bin++;
        }
        
        // 限制到精确的批量大小
        if (remove_batch.size() > batch_size) remove_batch.resize(batch_size);
        if (add_batch.size() > batch_size) add_batch.resize(batch_size);
        
        LOG("  Actual remove batch: " + to_string(remove_batch.size()) + " edges");
        LOG("  Actual add batch: " + to_string(add_batch.size()) + " edges");
        
        // 测试MCD/PCD
        LOG("\n  Testing MCD/PCD...");
        Graph mcdpcd_graph = initial_graph.copy();
        MCDPCDCore mcdpcd(mcdpcd_graph);
        
        Timer mcdpcd_timer;
        double mcdpcd_time = mcdpcd.process_batch(remove_batch, add_batch);
        
        // 检查是否超时
        if (mcdpcd_time < 0) {
            LOG("    MCD/PCD TIMEOUT (> 5 minutes)");
            mcdpcd_time = -1;
        } else {
            LOG("    MCD/PCD time: " + to_string(mcdpcd_time) + " ms");
        }
        
        // 测试UCR
        LOG("\n  Testing UCR...");
        Graph ucr_graph = initial_graph.copy();
        UCRBatch ucr(ucr_graph);
        
        Timer ucr_timer;
        double ucr_time = ucr.process_batch(remove_batch, add_batch);
        LOG("    UCR time: " + to_string(ucr_time) + " ms");
        
        // 计算加速比
        double speedup = -1;
        if (mcdpcd_time > 0) {
            speedup = mcdpcd_time / ucr_time;
            LOG("\n  Speedup: " + to_string(speedup) + "x");
        } else {
            LOG("\n  Speedup: N/A (MCD/PCD timeout)");
        }
        
        // 写入结果
        global_results << dataset_name << "\t" 
                      << batch_size << "\t"
                      << fixed << setprecision(2);
        
        if (mcdpcd_time > 0) {
            global_results << mcdpcd_time << "\t";
        } else {
            global_results << "TIMEOUT\t";
        }
        
        global_results << ucr_time << "\t";
        
        if (speedup > 0) {
            global_results << speedup << "\n";
        } else {
            global_results << "N/A\n";
        }
        
        global_results.flush();
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
        
        // 跳过隐藏文件和系统文件
        if (filename.empty() || filename[0] == '.') {
            continue;
        }
        
        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".txt") {
            string filepath = dir + "/" + filename;
            string name = filename.substr(0, filename.size() - 4);
            
            datasets.push_back({filepath, name});
        }
    }
    closedir(dirp);
    
    // 按文件名排序
    sort(datasets.begin(), datasets.end());
    
    return datasets;
}

int main() {
    LOG("===========================================");
    LOG("Experiment 3: UCR vs MCD/PCD Batch Size");
    LOG("Batch sizes: 1, 10, 100 edges");
    LOG("Timeout: 300 seconds (5 minutes)");
    LOG("Window: 50 bins out of 1000 bins");
    LOG("===========================================");
    
    // 打开全局结果文件
    string output_file = "/home/jding/e3_ucr_vs_mcdpcd_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    // 写入文件头
    global_results << "Dataset\tBatch_Size\tMCDPCD_Time(ms)\tUCR_Time(ms)\tSpeedup\n";
    global_results.flush();
    
    LOG("Results will be saved to: " + output_file);
    
    // 获取所有数据集
    string dataset_dir = "/home/jding/dataset";
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
            SlidingWindowBatchExperiment experiment(filepath, name);
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
    LOG("Experiment 3 completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}