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
#include <omp.h>

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
    
    size_t degree(vertex_t v) const {
        if (v >= num_vertices) return 0;
        return adj[v].size();
    }
    
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
};

// UCR基础类（所有UCR变体的基类）
class UCRBase {
protected:
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
    UCRBase(Graph& g) : G(g) {
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
    
    virtual double process_all(const vector<edge_t>& remove_edges, 
                              const vector<edge_t>& add_edges) = 0;
};

// 1. UCR-Basic：无优化版本，逐条处理
class UCRBasic : public UCRBase {
public:
    UCRBasic(Graph& g) : UCRBase(g) {}
    
    double process_all(const vector<edge_t>& remove_edges, 
                      const vector<edge_t>& add_edges) override {
        Timer timer;
        
        // 逐条处理删除
        for (const auto& edge : remove_edges) {
            process_removal(edge.first, edge.second);
        }
        
        // 逐条处理插入
        for (const auto& edge : add_edges) {
            process_insertion(edge.first, edge.second);
        }
        
        return timer.elapsed_milliseconds();
    }
    
private:
    void process_removal(vertex_t u, vertex_t v) {
        if (u >= G.num_vertices || v >= G.num_vertices) return;
        
        int ku = G.core[u];
        int kv = G.core[v];
        
        // 更新r值
        if (ku < kv) r_values[u]--;
        else if (ku > kv) r_values[v]--;
        
        G.remove_edge(u, v);
        
        // 处理降级
        vertex_t root = (ku <= kv) ? u : v;
        int k = G.core[root];
        
        if (k == 0) return;
        
        // 找受影响的子核
        queue<vertex_t> q;
        unordered_set<vertex_t> visited;
        q.push(root);
        visited.insert(root);
        
        vector<vertex_t> subcore;
        unordered_map<vertex_t, int> cd;
        
        while (!q.empty()) {
            vertex_t w = q.front();
            q.pop();
            
            if (G.core[w] == k) {
                subcore.push_back(w);
                
                int count = 0;
                for (vertex_t x : G.adj[w]) {
                    if (G.core[x] >= k) count++;
                }
                cd[w] = count;
                
                for (vertex_t x : G.adj[w]) {
                    if (G.core[x] == k && !visited.count(x)) {
                        visited.insert(x);
                        q.push(x);
                    }
                }
            }
        }
        
        // 级联降级
        queue<vertex_t> degrade_queue;
        unordered_set<vertex_t> degraded;
        
        for (vertex_t w : subcore) {
            if (cd[w] < k) {
                degrade_queue.push(w);
            }
        }
        
        while (!degrade_queue.empty()) {
            vertex_t w = degrade_queue.front();
            degrade_queue.pop();
            
            if (G.core[w] != k || degraded.count(w)) continue;
            
            G.core[w] = k - 1;
            degraded.insert(w);
            
            for (vertex_t x : G.adj[w]) {
                if (G.core[x] == k && !degraded.count(x)) {
                    cd[x]--;
                    if (cd[x] < k) {
                        degrade_queue.push(x);
                    }
                }
                
                if (G.core[x] > k - 1) {
                    r_values[x]--;
                }
            }
        }
        
        // 更新s值
        unordered_set<vertex_t> update_set;
        for (vertex_t w : degraded) {
            update_set.insert(w);
            for (vertex_t x : G.adj[w]) {
                update_set.insert(x);
            }
        }
        
        for (vertex_t w : update_set) {
            update_s_value(w);
        }
    }
    
    void process_insertion(vertex_t u, vertex_t v) {
        G.ensure_vertex(max(u, v));
        
        if (u >= r_values.size() || v >= r_values.size()) {
            size_t new_size = max(u, v) + 1;
            r_values.resize(new_size, 0);
            s_values.resize(new_size, 0);
        }
        
        if (G.adj[u].size() == 0) G.core[u] = 1;
        if (G.adj[v].size() == 0) G.core[v] = 1;
        
        G.add_edge(u, v);
        
        int ku = G.core[u];
        int kv = G.core[v];
        
        // 更新r值
        if (ku < kv) r_values[u]++;
        else if (ku > kv) r_values[v]++;
        
        // 更新s值
        unordered_set<vertex_t> affected;
        affected.insert(u);
        affected.insert(v);
        
        for (vertex_t w : affected) {
            update_s_value(w);
            for (vertex_t x : G.adj[w]) {
                if (G.core[x] == G.core[w]) {
                    update_s_value(x);
                }
            }
        }
        
        // 检查升级
        vertex_t root = (ku <= kv) ? u : v;
        int k = G.core[root];
        
        if (!is_qualified(root)) {
            return;
        }
        
        // 找候选集
        queue<vertex_t> q;
        unordered_set<vertex_t> visited;
        q.push(root);
        visited.insert(root);
        
        vector<vertex_t> candidates;
        unordered_map<vertex_t, int> cd;
        
        while (!q.empty()) {
            vertex_t w = q.front();
            q.pop();
            
            if (G.core[w] == k && is_qualified(w)) {
                candidates.push_back(w);
                cd[w] = r_values[w] + s_values[w];
                
                for (vertex_t x : G.adj[w]) {
                    if (G.core[x] == k && !visited.count(x)) {
                        visited.insert(x);
                        q.push(x);
                    }
                }
            }
        }
        
        // 模拟升级
        unordered_set<vertex_t> evicted;
        bool stable = false;
        
        while (!stable) {
            stable = true;
            vector<vertex_t> to_evict;
            
            for (vertex_t w : candidates) {
                if (evicted.count(w)) continue;
                if (cd[w] <= k) {
                    to_evict.push_back(w);
                    stable = false;
                }
            }
            
            for (vertex_t w : to_evict) {
                evicted.insert(w);
                
                for (vertex_t x : G.adj[w]) {
                    if (G.core[x] == k && !evicted.count(x) && 
                        find(candidates.begin(), candidates.end(), x) != candidates.end()) {
                        cd[x]--;
                    }
                }
            }
        }
        
        // 升级
        unordered_set<vertex_t> upgraded;
        for (vertex_t w : candidates) {
            if (!evicted.count(w)) {
                G.core[w]++;
                upgraded.insert(w);
            }
        }
        
        // 更新r值和s值
        for (vertex_t w : upgraded) {
            for (vertex_t x : G.adj[w]) {
                if (x < r_values.size() && G.core[x] < G.core[w]) {
                    r_values[x]++;
                }
            }
        }
        
        unordered_set<vertex_t> update_set;
        for (vertex_t w : upgraded) {
            update_set.insert(w);
            for (vertex_t x : G.adj[w]) {
                update_set.insert(x);
                for (vertex_t y : G.adj[x]) {
                    if (G.core[y] == G.core[x]) {
                        update_set.insert(y);
                    }
                }
            }
        }
        
        for (vertex_t w : update_set) {
            update_s_value(w);
        }
    }
};

// 2. UCR-Batch：只有批处理优化
class UCRBatch : public UCRBase {
public:
    UCRBatch(Graph& g) : UCRBase(g) {}
    
    double process_all(const vector<edge_t>& remove_edges, 
                      const vector<edge_t>& add_edges) override {
        Timer timer;
        
        // 批量处理删除
        batch_process_removals(remove_edges);
        
        // 批量处理插入
        batch_process_insertions(add_edges);
        
        return timer.elapsed_milliseconds();
    }
    
private:
    void batch_process_removals(const vector<edge_t>& edges) {
        // Phase 1: 批量更新r值和删除边
        unordered_map<int, unordered_set<vertex_t>> k_groups;
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv) r_values[u]--;
                else if (ku > kv) r_values[v]--;
                
                if (ku > 0) k_groups[ku].insert(u);
                if (kv > 0) k_groups[kv].insert(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // Phase 2: 按核心度层次处理
        vector<int> k_values;
        for (const auto& pair : k_groups) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            process_core_level_removal(k, k_groups[k]);
        }
    }
    
    void batch_process_insertions(const vector<edge_t>& edges) {
        // Phase 1: 批量添加边并更新r值
        unordered_set<vertex_t> affected_vertices;
        
        vertex_t max_vertex = 0;
        for (const auto& edge : edges) {
            max_vertex = max(max_vertex, max(edge.first, edge.second));
        }
        
        if (max_vertex >= G.num_vertices) {
            G.ensure_vertex(max_vertex);
            r_values.resize(G.num_vertices, 0);
            s_values.resize(G.num_vertices, 0);
        }
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
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
        
        // Phase 2: 批量更新s值
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
        
        // Phase 3: 批量处理升级
        unordered_set<vertex_t> all_candidates;
        for (vertex_t v : affected_vertices) {
            if (is_qualified(v)) {
                all_candidates.insert(v);
            }
        }
        
        if (!all_candidates.empty()) {
            vector<vector<vertex_t>> components;
            identify_components(all_candidates, components);
            
            for (const auto& component : components) {
                process_component_upgrade(component);
            }
        }
    }
    
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

// 滑动窗口实验类
class OptimizationExperiment {
private:
    string dataset_path;
    string dataset_name;
    vector<TemporalEdge> edges;
    vector<vector<edge_t>> bins;
    
    // 减小实验规模
    static constexpr int WINDOW_SIZE = 10;   // 减小到10个bins
    static constexpr int NUM_BINS = 100;     // 减小到100个bins
    static constexpr int NUM_SLIDES = 3;     // 只滑动3次
    
public:
    OptimizationExperiment(const string& path, const string& name) 
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
        
        // 选择起始位置
        int starting_bin = 40;
        if (starting_bin + WINDOW_SIZE > bins.size()) {
            starting_bin = bins.size() - WINDOW_SIZE - 10;
        }
        if (starting_bin < 0) starting_bin = 0;
        
        // 构建初始窗口
        LOG("Building initial window...");
        Graph initial_graph;
        size_t total_edges = 0;
        
        for (int i = starting_bin; i < starting_bin + WINDOW_SIZE; ++i) {
            total_edges += bins[i].size();
            for (const auto& edge : bins[i]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
        }
        
        LOG("  Initial edges: " + to_string(total_edges));
        LOG("  Initial vertices: " + to_string(initial_graph.num_vertices));
        
        // 计算初始核心度
        Timer init_timer;
        initial_graph.compute_core_numbers_bz();
        LOG("  Initial core computation: " + to_string(init_timer.elapsed_milliseconds()) + " ms");
        
        // 运行实验
        run_optimization_comparison(initial_graph, starting_bin);
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
        size_t max_lines = 500000;  // 限制最多读取50万行
        
        while (getline(file, line) && line_count < max_lines) {
            line_count++;
            if (line_count % 100000 == 0) {
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
    
    void run_optimization_comparison(Graph& initial_graph, int starting_bin) {
        LOG("\n--- Running optimization comparison ---");
        
        // 记录各算法的时间
        vector<double> basic_times, batch_times;
        
        // 滑动NUM_SLIDES次
        for (int slide = 0; slide < NUM_SLIDES; ++slide) {
            int remove_bin = starting_bin + slide;
            int add_bin = starting_bin + WINDOW_SIZE + slide;
            
            if (add_bin >= bins.size()) {
                LOG("Reached end of bins at slide " + to_string(slide + 1));
                break;
            }
            
            LOG("\nSlide " + to_string(slide + 1) + "/" + to_string(NUM_SLIDES) + ":");
            LOG("  Remove bin " + to_string(remove_bin) + ": " + 
                to_string(bins[remove_bin].size()) + " edges");
            LOG("  Add bin " + to_string(add_bin) + ": " + 
                to_string(bins[add_bin].size()) + " edges");
            
            // 1. UCR-Basic
            {
                Graph ucr_graph = initial_graph.copy();
                UCRBasic ucr(ucr_graph);
                Timer timer;
                double time = ucr.process_all(bins[remove_bin], bins[add_bin]);
                basic_times.push_back(time);
                LOG("    UCR-Basic: " + to_string(time) + " ms");
            }
            
            // 2. UCR-Batch
            {
                Graph ucr_graph = initial_graph.copy();
                UCRBatch ucr(ucr_graph);
                Timer timer;
                double time = ucr.process_all(bins[remove_bin], bins[add_bin]);
                batch_times.push_back(time);
                LOG("    UCR-Batch: " + to_string(time) + " ms");
            }
            
            // 更新初始图为下一次滑动准备
            for (const auto& edge : bins[remove_bin]) {
                initial_graph.remove_edge(edge.first, edge.second);
            }
            for (const auto& edge : bins[add_bin]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
            initial_graph.compute_core_numbers_bz();
        }
        
        // 计算平均值
        if (!basic_times.empty()) {
            double avg_basic = 0, avg_batch = 0;
            
            for (size_t i = 0; i < basic_times.size(); ++i) {
                avg_basic += basic_times[i];
                avg_batch += batch_times[i];
            }
            
            avg_basic /= basic_times.size();
            avg_batch /= basic_times.size();
            
            LOG("\nAverage times over " + to_string(basic_times.size()) + " slides:");
            LOG("  UCR-Basic: " + to_string(avg_basic) + " ms");
            LOG("  UCR-Batch: " + to_string(avg_batch) + " ms");
            
            LOG("\nOptimization contribution:");
            LOG("  Batch speedup: " + to_string(avg_basic / avg_batch) + "x");
            
            // 写入结果
            global_results << dataset_name << "\t"
                          << fixed << setprecision(2)
                          << avg_basic << "\t"
                          << avg_batch << "\t"
                          << (avg_basic / avg_batch) << "\n";
            global_results.flush();
        }
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
    LOG("Experiment 4: UCR Optimization Analysis");
    LOG("Window: 10 bins, Total: 100 bins, Slides: 3");
    LOG("Simplified version: Basic vs Batch only");
    LOG("===========================================");
    
    // 打开全局结果文件
    string output_file = "/home/jding/e4_ucr_optimization_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    // 写入文件头
    global_results << "Dataset\tBasic(ms)\tBatch(ms)\tBatch_Speedup\n";
    global_results.flush();
    
    LOG("Results will be saved to: " + output_file);
    
    // 获取所有数据集
    string dataset_dir = "/home/jding/dataset";
    auto datasets = get_datasets(dataset_dir);
    
    LOG("\nFound " + to_string(datasets.size()) + " datasets");
    
    // 只处理前5个数据集作为测试
    int max_datasets = min(5, (int)datasets.size());
    LOG("Processing first " + to_string(max_datasets) + " datasets for testing");
    
    // 对每个数据集运行实验
    for (int i = 0; i < max_datasets; ++i) {
        const string& filepath = datasets[i].first;
        const string& name = datasets[i].second;
        
        LOG("\n[" + to_string(i+1) + "/" + to_string(max_datasets) + "] " + name);
        
        try {
            OptimizationExperiment experiment(filepath, name);
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
    LOG("Experiment 4 completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}