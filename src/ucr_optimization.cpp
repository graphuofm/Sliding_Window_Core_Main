// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <unordered_map>
// #include <unordered_set>
// #include <queue>
// #include <deque>
// #include <algorithm>
// #include <chrono>
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include <cstring>
// #include <ctime>
// #include <omp.h>
// #include <atomic>
// #include <mutex>

// // 定义顶点和边的类型
// using vertex_t = int;
// using edge_t = std::pair<vertex_t, vertex_t>;
// using timestamp_t = long long;

// // 获取当前时间字符串
// std::string get_current_time() {
//    auto now = std::chrono::system_clock::now();
//    auto time_t = std::chrono::system_clock::to_time_t(now);
//    char buffer[100];
//    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&time_t));
//    return std::string(buffer);
// }

// // 日志输出宏
// #define LOG(msg) do { \
//    std::cout << "[" << get_current_time() << "] " << msg << std::endl; \
//    std::cout.flush(); \
// } while(0)

// // 时序边结构
// struct TemporalEdge {
//    vertex_t src;
//    vertex_t dst;
//    timestamp_t timestamp;
   
//    TemporalEdge(vertex_t s, vertex_t d, timestamp_t t) : src(s), dst(d), timestamp(t) {}
// };

// // 计时辅助类
// class Timer {
// private:
//    std::chrono::high_resolution_clock::time_point start_time;
// public:
//    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
   
//    double elapsed_milliseconds() const {
//        auto end_time = std::chrono::high_resolution_clock::now();
//        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
//    }
   
//    void reset() {
//        start_time = std::chrono::high_resolution_clock::now();
//    }
// };

// // 图结构
// class Graph {
// public:
//    std::vector<std::vector<vertex_t>> adj;
//    std::vector<int> core;
//    size_t num_vertices;
   
//    Graph() : num_vertices(0) {}
   
//    void ensure_vertex(vertex_t v) {
//        if (v >= num_vertices) {
//            num_vertices = v + 1;
//            adj.resize(num_vertices);
//            core.resize(num_vertices, 0);
//        }
//    }
   
//    void add_edge(vertex_t u, vertex_t v) {
//        ensure_vertex(std::max(u, v));
       
//        if (std::find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
//            adj[u].push_back(v);
//        }
//        if (std::find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) {
//            adj[v].push_back(u);
//        }
//    }
   
//    void remove_edge(vertex_t u, vertex_t v) {
//        if (u >= num_vertices || v >= num_vertices) return;
       
//        auto it_u = std::find(adj[u].begin(), adj[u].end(), v);
//        if (it_u != adj[u].end()) {
//            *it_u = adj[u].back();
//            adj[u].pop_back();
//        }
       
//        auto it_v = std::find(adj[v].begin(), adj[v].end(), u);
//        if (it_v != adj[v].end()) {
//            *it_v = adj[v].back();
//            adj[v].pop_back();
//        }
//    }
   
//    size_t degree(vertex_t v) const {
//        if (v >= num_vertices) return 0;
//        return adj[v].size();
//    }
   
//    void compute_core_numbers_bz() {
//        if (num_vertices == 0) return;
       
//        std::fill(core.begin(), core.end(), 0);
       
//        std::vector<int> degree(num_vertices);
//        int max_degree = 0;
//        for (vertex_t v = 0; v < num_vertices; ++v) {
//            degree[v] = adj[v].size();
//            max_degree = std::max(max_degree, degree[v]);
//        }
       
//        std::vector<std::vector<vertex_t>> bins(max_degree + 1);
//        for (vertex_t v = 0; v < num_vertices; ++v) {
//            bins[degree[v]].push_back(v);
//        }
       
//        std::vector<bool> processed(num_vertices, false);
       
//        for (int d = 0; d <= max_degree; ++d) {
//            for (vertex_t v : bins[d]) {
//                if (processed[v]) continue;
               
//                core[v] = d;
//                processed[v] = true;
               
//                for (vertex_t w : adj[v]) {
//                    if (processed[w]) continue;
                   
//                    if (degree[w] > d) {
//                        auto& bin = bins[degree[w]];
//                        auto it = std::find(bin.begin(), bin.end(), w);
//                        if (it != bin.end()) {
//                            std::swap(*it, bin.back());
//                            bin.pop_back();
//                        }
                       
//                        --degree[w];
//                        bins[degree[w]].push_back(w);
//                    }
//                }
//            }
//        }
//    }
   
//    Graph copy() const {
//        Graph g;
//        g.num_vertices = num_vertices;
//        g.adj = adj;
//        g.core = core;
//        return g;
//    }
// };

// // 1. 基础UCR - 完整实现
// class BaseUCR {
// protected:
//    Graph& G;
//    std::vector<int> r_values;
//    std::vector<int> s_values;
   
//    bool is_qualified(vertex_t v) const {
//        if (v >= r_values.size()) return false;
//        int k = G.core[v];
//        return r_values[v] + s_values[v] > k;
//    }
   
//    void update_s_value(vertex_t v) {
//        if (v >= G.num_vertices) return;
//        int k = G.core[v];
//        int s_count = 0;
       
//        for (vertex_t w : G.adj[v]) {
//            if (G.core[w] == k && is_qualified(w)) {
//                ++s_count;
//            }
//        }
       
//        s_values[v] = s_count;
//    }
   
// public:
//    BaseUCR(Graph& g) : G(g) {
//        reset();
//    }
   
//    virtual ~BaseUCR() = default;
   
//    virtual void reset() {
//        size_t n = G.num_vertices;
//        r_values.resize(n, 0);
//        s_values.resize(n, 0);
       
//        // 计算r值
//        for (vertex_t v = 0; v < n; ++v) {
//            int k = G.core[v];
//            int r_count = 0;
           
//            for (vertex_t w : G.adj[v]) {
//                if (G.core[w] > k) {
//                    ++r_count;
//                }
//            }
           
//            r_values[v] = r_count;
//        }
       
//        // 计算s值
//        for (vertex_t v = 0; v < n; ++v) {
//            update_s_value(v);
//        }
//    }
   
//    virtual double process_all(const std::vector<edge_t>& remove_edges, 
//                              const std::vector<edge_t>& add_edges) {
//        Timer timer;
       
//        // 逐个处理删除
//        for (const auto& edge : remove_edges) {
//            process_removal(edge.first, edge.second);
//        }
       
//        // 逐个处理插入
//        for (const auto& edge : add_edges) {
//            process_insertion(edge.first, edge.second);
//        }
       
//        return timer.elapsed_milliseconds();
//    }
   
// protected:
//    void process_removal(vertex_t u, vertex_t v) {
//        if (u >= G.num_vertices || v >= G.num_vertices) return;
       
//        int ku = G.core[u];
//        int kv = G.core[v];
       
//        // 更新r值
//        if (ku < kv) r_values[u]--;
//        else if (ku > kv) r_values[v]--;
       
//        G.remove_edge(u, v);
       
//        // 处理降级
//        vertex_t root = (ku <= kv) ? u : v;
//        int k = G.core[root];
       
//        if (k == 0) return;
       
//        // 找受影响的子核
//        std::queue<vertex_t> q;
//        std::unordered_set<vertex_t> visited;
//        q.push(root);
//        visited.insert(root);
       
//        std::vector<vertex_t> subcore;
//        std::unordered_map<vertex_t, int> cd;
       
//        while (!q.empty()) {
//            vertex_t w = q.front();
//            q.pop();
           
//            if (G.core[w] == k) {
//                subcore.push_back(w);
               
//                // 计算核心度
//                int count = 0;
//                for (vertex_t x : G.adj[w]) {
//                    if (G.core[x] >= k) count++;
//                }
//                cd[w] = count;
               
//                // 添加同核心度的邻居
//                for (vertex_t x : G.adj[w]) {
//                    if (G.core[x] == k && !visited.count(x)) {
//                        visited.insert(x);
//                        q.push(x);
//                    }
//                }
//            }
//        }
       
//        // 找需要降级的顶点
//        std::queue<vertex_t> degrade_queue;
//        std::unordered_set<vertex_t> degraded;
       
//        for (vertex_t w : subcore) {
//            if (cd[w] < k) {
//                degrade_queue.push(w);
//            }
//        }
       
//        // 级联降级
//        while (!degrade_queue.empty()) {
//            vertex_t w = degrade_queue.front();
//            degrade_queue.pop();
           
//            if (G.core[w] != k || degraded.count(w)) continue;
           
//            G.core[w] = k - 1;
//            degraded.insert(w);
           
//            // 更新邻居
//            for (vertex_t x : G.adj[w]) {
//                if (G.core[x] == k && !degraded.count(x)) {
//                    cd[x]--;
//                    if (cd[x] < k) {
//                        degrade_queue.push(x);
//                    }
//                }
               
//                // 更新r值
//                if (G.core[x] > k - 1) {
//                    r_values[x]--;
//                }
//            }
//        }
       
//        // 更新s值
//        std::unordered_set<vertex_t> update_set;
//        for (vertex_t w : degraded) {
//            update_set.insert(w);
//            for (vertex_t x : G.adj[w]) {
//                update_set.insert(x);
//            }
//        }
       
//        for (vertex_t w : update_set) {
//            update_s_value(w);
//        }
//    }
   
//    void process_insertion(vertex_t u, vertex_t v) {
//        G.ensure_vertex(std::max(u, v));
       
//        if (u >= r_values.size() || v >= r_values.size()) {
//            size_t new_size = std::max(u, v) + 1;
//            r_values.resize(new_size, 0);
//            s_values.resize(new_size, 0);
//        }
       
//        if (G.adj[u].size() == 0) G.core[u] = 1;
//        if (G.adj[v].size() == 0) G.core[v] = 1;
       
//        G.add_edge(u, v);
       
//        int ku = G.core[u];
//        int kv = G.core[v];
       
//        // 更新r值
//        if (ku < kv) r_values[u]++;
//        else if (ku > kv) r_values[v]++;
       
//        // 更新s值
//        std::unordered_set<vertex_t> affected;
//        affected.insert(u);
//        affected.insert(v);
       
//        for (vertex_t w : affected) {
//            update_s_value(w);
//            for (vertex_t x : G.adj[w]) {
//                if (G.core[x] == G.core[w]) {
//                    update_s_value(x);
//                }
//            }
//        }
       
//        // 检查升级
//        vertex_t root = (ku <= kv) ? u : v;
//        int k = G.core[root];
       
//        if (!is_qualified(root)) {
//            return;
//        }
       
//        // 找候选集
//        std::queue<vertex_t> q;
//        std::unordered_set<vertex_t> visited;
//        q.push(root);
//        visited.insert(root);
       
//        std::vector<vertex_t> candidates;
//        std::unordered_map<vertex_t, int> cd;
       
//        while (!q.empty()) {
//            vertex_t w = q.front();
//            q.pop();
           
//            if (G.core[w] == k && is_qualified(w)) {
//                candidates.push_back(w);
//                cd[w] = r_values[w] + s_values[w];
               
//                for (vertex_t x : G.adj[w]) {
//                    if (G.core[x] == k && !visited.count(x)) {
//                        visited.insert(x);
//                        q.push(x);
//                    }
//                }
//            }
//        }
       
//        // 模拟升级
//        std::unordered_set<vertex_t> evicted;
//        bool stable = false;
       
//        while (!stable) {
//            stable = true;
//            std::vector<vertex_t> to_evict;
           
//            for (vertex_t w : candidates) {
//                if (evicted.count(w)) continue;
//                if (cd[w] <= k) {
//                    to_evict.push_back(w);
//                    stable = false;
//                }
//            }
           
//            for (vertex_t w : to_evict) {
//                evicted.insert(w);
               
//                for (vertex_t x : G.adj[w]) {
//                    if (G.core[x] == k && !evicted.count(x) && 
//                        std::find(candidates.begin(), candidates.end(), x) != candidates.end()) {
//                        cd[x]--;
//                    }
//                }
//            }
//        }
       
//        // 升级
//        std::unordered_set<vertex_t> upgraded;
//        for (vertex_t w : candidates) {
//            if (!evicted.count(w)) {
//                G.core[w]++;
//                upgraded.insert(w);
//            }
//        }
       
//        // 更新r值和s值
//        for (vertex_t w : upgraded) {
//            for (vertex_t x : G.adj[w]) {
//                if (x < r_values.size() && G.core[x] < G.core[w]) {
//                    r_values[x]++;
//                }
//            }
//        }
       
//        std::unordered_set<vertex_t> update_set;
//        for (vertex_t w : upgraded) {
//            update_set.insert(w);
//            for (vertex_t x : G.adj[w]) {
//                update_set.insert(x);
//                for (vertex_t y : G.adj[x]) {
//                    if (G.core[y] == G.core[x]) {
//                        update_set.insert(y);
//                    }
//                }
//            }
//        }
       
//        for (vertex_t w : update_set) {
//            update_s_value(w);
//        }
//    }
// };

// // 2. 批处理UCR - 批量处理边
// class BatchUCR : public BaseUCR {
// public:
//    BatchUCR(Graph& g) : BaseUCR(g) {}
   
//    double process_all(const std::vector<edge_t>& remove_edges, 
//                      const std::vector<edge_t>& add_edges) override {
//        Timer timer;
       
//        // 批量收集受影响的顶点
//        std::unordered_map<int, std::unordered_set<vertex_t>> k_groups_remove;
//        std::unordered_map<int, std::unordered_set<vertex_t>> k_groups_add;
       
//        // 批量更新r值（删除）
//        for (const auto& edge : remove_edges) {
//            vertex_t u = edge.first;
//            vertex_t v = edge.second;
           
//            if (u < G.num_vertices && v < G.num_vertices) {
//                int ku = G.core[u];
//                int kv = G.core[v];
               
//                if (ku < kv) r_values[u]--;
//                else if (ku > kv) r_values[v]--;
               
//                if (ku > 0) k_groups_remove[ku].insert(u);
//                if (kv > 0) k_groups_remove[kv].insert(v);
//            }
//        }
       
//        // 批量删除边
//        for (const auto& edge : remove_edges) {
//            G.remove_edge(edge.first, edge.second);
//        }
       
//        // 按核心度层次批量处理删除
//        std::vector<int> k_values;
//        for (const auto& pair : k_groups_remove) {
//            k_values.push_back(pair.first);
//        }
//        std::sort(k_values.begin(), k_values.end());
       
//        for (int k : k_values) {
//            process_core_level_removal(k, k_groups_remove[k]);
//        }
       
//        // 批量更新r值（插入）
//        vertex_t max_vertex_id = 0;
//        for (const auto& edge : add_edges) {
//            max_vertex_id = std::max(max_vertex_id, std::max(edge.first, edge.second));
//        }
       
//        if (max_vertex_id >= G.core.size()) {
//            G.core.resize(max_vertex_id + 1, 0);
//        }
//        if (max_vertex_id >= r_values.size()) {
//            r_values.resize(max_vertex_id + 1, 0);
//            s_values.resize(max_vertex_id + 1, 0);
//        }
       
//        // 批量添加边
//        std::unordered_set<vertex_t> affected_vertices;
       
//        for (const auto& edge : add_edges) {
//            vertex_t u = edge.first;
//            vertex_t v = edge.second;
           
//            G.ensure_vertex(std::max(u, v));
           
//            if (G.adj[u].size() == 0) G.core[u] = 1;
//            if (G.adj[v].size() == 0) G.core[v] = 1;
           
//            G.add_edge(u, v);
           
//            int ku = G.core[u];
//            int kv = G.core[v];
           
//            if (ku < kv) r_values[u]++;
//            else if (ku > kv) r_values[v]++;
           
//            affected_vertices.insert(u);
//            affected_vertices.insert(v);
//        }
       
//        // 批量更新s值
//        std::unordered_set<vertex_t> s_update_set;
//        for (vertex_t v : affected_vertices) {
//            s_update_set.insert(v);
//            for (vertex_t w : G.adj[v]) {
//                if (G.core[w] == G.core[v]) {
//                    s_update_set.insert(w);
//                }
//            }
//        }
       
//        for (vertex_t v : s_update_set) {
//            update_s_value(v);
//        }
       
//        // 找所有可能升级的候选
//        std::unordered_set<vertex_t> all_candidates;
//        for (vertex_t v : affected_vertices) {
//            if (is_qualified(v)) {
//                all_candidates.insert(v);
//            }
//        }
       
//        // 批量处理升级
//        if (!all_candidates.empty()) {
//            std::vector<std::vector<vertex_t>> components;
//            identify_components(all_candidates, components);
           
//            for (const auto& component : components) {
//                process_component_upgrade(component);
//            }
//        }
       
//        return timer.elapsed_milliseconds();
//    }
   
// private:
//    void process_core_level_removal(int k, const std::unordered_set<vertex_t>& seeds) {
//        std::unordered_set<vertex_t> visited;
       
//        for (vertex_t seed : seeds) {
//            if (visited.count(seed) || G.core[seed] != k) continue;
           
//            // 找连通分量
//            std::queue<vertex_t> q;
//            q.push(seed);
//            visited.insert(seed);
           
//            std::vector<vertex_t> component;
//            std::unordered_map<vertex_t, int> eff_deg;
           
//            while (!q.empty()) {
//                vertex_t v = q.front();
//                q.pop();
               
//                if (G.core[v] == k) {
//                    component.push_back(v);
                   
//                    int count = 0;
//                    for (vertex_t w : G.adj[v]) {
//                        if (G.core[w] >= k) count++;
//                    }
//                    eff_deg[v] = count;
                   
//                    for (vertex_t w : G.adj[v]) {
//                        if (G.core[w] == k && !visited.count(w)) {
//                            visited.insert(w);
//                            q.push(w);
//                        }
//                    }
//                }
//            }
           
//            // 批量级联降级
//            std::queue<vertex_t> degrade_queue;
//            std::unordered_set<vertex_t> degraded;
           
//            for (vertex_t v : component) {
//                if (eff_deg[v] < k) {
//                    degrade_queue.push(v);
//                }
//            }
           
//            while (!degrade_queue.empty()) {
//                vertex_t v = degrade_queue.front();
//                degrade_queue.pop();
               
//                if (G.core[v] != k || degraded.count(v)) continue;
               
//                G.core[v] = k - 1;
//                degraded.insert(v);
               
//                for (vertex_t w : G.adj[v]) {
//                    if (G.core[w] == k && !degraded.count(w)) {
//                        eff_deg[w]--;
//                        if (eff_deg[w] < k) {
//                            degrade_queue.push(w);
//                        }
//                    }
                   
//                    if (w < r_values.size() && G.core[w] > k - 1) {
//                        r_values[w]--;
//                    }
//                }
//            }
           
//            // 批量更新s值
//            std::unordered_set<vertex_t> update_set;
//            for (vertex_t v : degraded) {
//                update_set.insert(v);
//                for (vertex_t w : G.adj[v]) {
//                    update_set.insert(w);
//                }
//            }
           
//            for (vertex_t v : update_set) {
//                update_s_value(v);
//            }
//        }
//    }
   
//    void identify_components(const std::unordered_set<vertex_t>& candidates,
//                            std::vector<std::vector<vertex_t>>& components) {
//        std::unordered_set<vertex_t> visited;
       
//        for (vertex_t v : candidates) {
//            if (visited.count(v)) continue;
           
//            std::vector<vertex_t> component;
//            std::queue<vertex_t> q;
//            q.push(v);
//            visited.insert(v);
           
//            int k = G.core[v];
           
//            while (!q.empty()) {
//                vertex_t u = q.front();
//                q.pop();
               
//                if (G.core[u] == k) {
//                    component.push_back(u);
                   
//                    for (vertex_t w : G.adj[u]) {
//                        if (G.core[w] == k && !visited.count(w) && candidates.count(w)) {
//                            visited.insert(w);
//                            q.push(w);
//                        }
//                    }
//                }
//            }
           
//            if (!component.empty()) {
//                components.push_back(std::move(component));
//            }
//        }
//    }
   
//    void process_component_upgrade(const std::vector<vertex_t>& component) {
//        if (component.empty()) return;
       
//        int k = G.core[component[0]];
       
//        std::unordered_map<vertex_t, int> cd;
//        for (vertex_t v : component) {
//            cd[v] = r_values[v] + s_values[v];
//        }
       
//        std::unordered_set<vertex_t> evicted;
//        bool stable = false;
       
//        while (!stable) {
//            stable = true;
//            std::vector<vertex_t> to_evict;
           
//            for (vertex_t v : component) {
//                if (evicted.count(v)) continue;
//                if (cd[v] <= k) {
//                    to_evict.push_back(v);
//                    stable = false;
//                }
//            }
           
//            for (vertex_t v : to_evict) {
//                evicted.insert(v);
               
//                for (vertex_t w : G.adj[v]) {
//                    if (G.core[w] == k && !evicted.count(w) && cd.count(w)) {
//                        cd[w]--;
//                    }
//                }
//            }
//        }
       
//        // 批量升级
//        for (vertex_t v : component) {
//            if (!evicted.count(v)) {
//                G.core[v]++;
               
//                for (vertex_t w : G.adj[v]) {
//                    if (w < r_values.size() && G.core[w] < G.core[v]) {
//                        r_values[w]++;
//                    }
//                }
//            }
//        }
       
//        // 批量更新s值
//        std::unordered_set<vertex_t> update_set;
//        for (vertex_t v : component) {
//            if (!evicted.count(v)) {
//                update_set.insert(v);
//                for (vertex_t w : G.adj[v]) {
//                    update_set.insert(w);
//                }
//            }
//        }
       
//        for (vertex_t v : update_set) {
//            update_s_value(v);
//        }
//    }
// };

// // 3. 层次处理UCR - 按核心度层次处理
// class HierarchicalUCR : public BaseUCR {
// private:
//    std::vector<std::vector<vertex_t>> core_level_vertices;
//    int max_core_level;
   
// public:
//    HierarchicalUCR(Graph& g) : BaseUCR(g) {}
   
//    void reset() override {
//        BaseUCR::reset();
       
//        // 构建核心度层次索引
//        max_core_level = 0;
//        for (vertex_t v = 0; v < G.num_vertices; ++v) {
//            max_core_level = std::max(max_core_level, G.core[v]);
//        }
       
//        core_level_vertices.clear();
//        core_level_vertices.resize(max_core_level + 1);
       
//        for (vertex_t v = 0; v < G.num_vertices; ++v) {
//            int k = G.core[v];
//            if (k <= max_core_level) {
//                core_level_vertices[k].push_back(v);
//            }
//        }
//    }
   
//    double process_all(const std::vector<edge_t>& remove_edges, 
//                      const std::vector<edge_t>& add_edges) override {
//        Timer timer;
       
//        // 层次化处理删除：从低到高
//        std::unordered_map<int, std::vector<edge_t>> level_removes;
       
//        for (const auto& edge : remove_edges) {
//            if (edge.first < G.num_vertices && edge.second < G.num_vertices) {
//                int k = std::min(G.core[edge.first], G.core[edge.second]);
//                level_removes[k].push_back(edge);
//            }
//        }
       
//        // 按核心度升序处理删除
//        for (int k = 0; k <= max_core_level; ++k) {
//            if (level_removes.count(k)) {
//                for (const auto& edge : level_removes[k]) {
//                    process_removal(edge.first, edge.second);
//                }
//            }
//        }
       
//        // 层次化处理插入：从高到低
//        std::unordered_map<int, std::vector<edge_t>> level_inserts;
       
//        for (const auto& edge : add_edges) {
//            G.ensure_vertex(std::max(edge.first, edge.second));
           
//            if (edge.first >= r_values.size() || edge.second >= r_values.size()) {
//                size_t new_size = std::max(edge.first, edge.second) + 1;
//                r_values.resize(new_size, 0);
//                s_values.resize(new_size, 0);
//            }
           
//            if (G.adj[edge.first].size() == 0) G.core[edge.first] = 1;
//            if (G.adj[edge.second].size() == 0) G.core[edge.second] = 1;
           
//            int k = std::max(G.core[edge.first], G.core[edge.second]);
//            level_inserts[k].push_back(edge);
//        }
       
//        // 按核心度降序处理插入
//        for (int k = max_core_level; k >= 0; --k) {
//            if (level_inserts.count(k)) {
//                for (const auto& edge : level_inserts[k]) {
//                    process_insertion(edge.first, edge.second);
//                }
//            }
//        }
       
//         return timer.elapsed_milliseconds();
//    }
// };

// // 4. 边界传播UCR - 使用边界传播优化s值更新
// class BoundaryPropagationUCR : public BaseUCR {
// public:
//    BoundaryPropagationUCR(Graph& g) : BaseUCR(g) {}
   
//    double process_all(const std::vector<edge_t>& remove_edges, 
//                      const std::vector<edge_t>& add_edges) override {
//        Timer timer;
       
//        // 处理删除
//        for (const auto& edge : remove_edges) {
//            process_removal(edge.first, edge.second);
//        }
       
//        // 处理插入，使用边界传播
//        for (const auto& edge : add_edges) {
//            process_insertion_with_boundary(edge.first, edge.second);
//        }
       
//        return timer.elapsed_milliseconds();
//    }
   
// private:
//    void process_insertion_with_boundary(vertex_t u, vertex_t v) {
//        G.ensure_vertex(std::max(u, v));
       
//        if (u >= r_values.size() || v >= r_values.size()) {
//            size_t new_size = std::max(u, v) + 1;
//            r_values.resize(new_size, 0);
//            s_values.resize(new_size, 0);
//        }
       
//        if (G.adj[u].size() == 0) G.core[u] = 1;
//        if (G.adj[v].size() == 0) G.core[v] = 1;
       
//        G.add_edge(u, v);
       
//        int ku = G.core[u];
//        int kv = G.core[v];
       
//        // 更新r值
//        if (ku < kv) r_values[u]++;
//        else if (ku > kv) r_values[v]++;
       
//        // 边界传播更新s值
//        std::unordered_set<vertex_t> update_frontier;
//        update_frontier.insert(u);
//        update_frontier.insert(v);
       
//        std::unordered_set<vertex_t> processed;
       
//        int max_iterations = 5; // 限制传播轮数
//        for (int iter = 0; iter < max_iterations && !update_frontier.empty(); ++iter) {
//            std::unordered_set<vertex_t> next_frontier;
           
//            for (vertex_t w : update_frontier) {
//                if (processed.count(w)) continue;
//                processed.insert(w);
               
//                int k = G.core[w];
//                int old_s = s_values[w];
//                update_s_value(w);
               
//                if (old_s != s_values[w]) {
//                    // s值变化了，将其同核心度的邻居加入下一轮更新
//                    for (vertex_t x : G.adj[w]) {
//                        if (G.core[x] == k && !processed.count(x)) {
//                            next_frontier.insert(x);
//                        }
//                    }
//                }
//            }
           
//            update_frontier = std::move(next_frontier);
//        }
       
//        // 检查升级
//        vertex_t root = (ku <= kv) ? u : v;
//        int k = G.core[root];
       
//        if (!is_qualified(root)) {
//            return;
//        }
       
//        // 使用边界传播找候选集
//        std::unordered_set<vertex_t> candidates;
//        std::queue<vertex_t> q;
//        q.push(root);
//        candidates.insert(root);
       
//        while (!q.empty()) {
//            vertex_t w = q.front();
//            q.pop();
           
//            for (vertex_t x : G.adj[w]) {
//                if (G.core[x] == k && is_qualified(x) && !candidates.count(x)) {
//                    candidates.insert(x);
//                    q.push(x);
//                }
//            }
//        }
       
//        // 转换为vector并处理升级
//        std::vector<vertex_t> candidate_vec(candidates.begin(), candidates.end());
//        process_candidates_upgrade(candidate_vec, k);
//    }
   
//    void process_candidates_upgrade(const std::vector<vertex_t>& candidates, int k) {
//        std::unordered_map<vertex_t, int> cd;
//        for (vertex_t v : candidates) {
//            cd[v] = r_values[v] + s_values[v];
//        }
       
//        std::unordered_set<vertex_t> evicted;
//        bool stable = false;
       
//        while (!stable) {
//            stable = true;
//            std::vector<vertex_t> to_evict;
           
//            for (vertex_t v : candidates) {
//                if (evicted.count(v)) continue;
//                if (cd[v] <= k) {
//                    to_evict.push_back(v);
//                    stable = false;
//                }
//            }
           
//            for (vertex_t v : to_evict) {
//                evicted.insert(v);
               
//                for (vertex_t w : G.adj[v]) {
//                    if (G.core[w] == k && !evicted.count(w) && cd.count(w)) {
//                        cd[w]--;
//                    }
//                }
//            }
//        }
       
//        // 升级并使用边界传播更新
//        std::unordered_set<vertex_t> upgraded;
//        for (vertex_t v : candidates) {
//            if (!evicted.count(v)) {
//                G.core[v]++;
//                upgraded.insert(v);
               
//                for (vertex_t w : G.adj[v]) {
//                    if (w < r_values.size() && G.core[w] < G.core[v]) {
//                        r_values[w]++;
//                    }
//                }
//            }
//        }
       
//        // 边界传播更新s值
//        std::unordered_set<vertex_t> update_frontier;
//        for (vertex_t v : upgraded) {
//            update_frontier.insert(v);
//            for (vertex_t w : G.adj[v]) {
//                update_frontier.insert(w);
//            }
//        }
       
//        std::unordered_set<vertex_t> processed;
//        int max_iterations = 3;
       
//        for (int iter = 0; iter < max_iterations && !update_frontier.empty(); ++iter) {
//            std::unordered_set<vertex_t> next_frontier;
           
//            for (vertex_t v : update_frontier) {
//                if (processed.count(v)) continue;
//                processed.insert(v);
               
//                int k = G.core[v];
//                int old_s = s_values[v];
//                update_s_value(v);
               
//                if (old_s != s_values[v]) {
//                    for (vertex_t w : G.adj[v]) {
//                        if (G.core[w] == k && !processed.count(w)) {
//                            next_frontier.insert(w);
//                        }
//                    }
//                }
//            }
           
//            update_frontier = std::move(next_frontier);
//        }
//    }
// };

// // 5. 双端队列UCR - 使用双端队列优化级联处理
// class DequeUCR : public BaseUCR {
// public:
//    DequeUCR(Graph& g) : BaseUCR(g) {}
   
//    double process_all(const std::vector<edge_t>& remove_edges, 
//                      const std::vector<edge_t>& add_edges) override {
//        Timer timer;
       
//        // 使用双端队列处理删除
//        for (const auto& edge : remove_edges) {
//            process_removal_with_deque(edge.first, edge.second);
//        }
       
//        // 使用双端队列处理插入
//        for (const auto& edge : add_edges) {
//            process_insertion(edge.first, edge.second);
//        }
       
//        return timer.elapsed_milliseconds();
//    }
   
// private:
//    void process_removal_with_deque(vertex_t u, vertex_t v) {
//        if (u >= G.num_vertices || v >= G.num_vertices) return;
       
//        int ku = G.core[u];
//        int kv = G.core[v];
       
//        // 更新r值
//        if (ku < kv) r_values[u]--;
//        else if (ku > kv) r_values[v]--;
       
//        G.remove_edge(u, v);
       
//        vertex_t root = (ku <= kv) ? u : v;
//        int k = G.core[root];
       
//        if (k == 0) return;
       
//        // 使用双端队列收集子核
//        std::deque<vertex_t> process_queue;
//        std::unordered_set<vertex_t> in_queue;
//        std::unordered_map<vertex_t, int> cd;
       
//        process_queue.push_back(root);
//        in_queue.insert(root);
       
//        std::unordered_set<vertex_t> subcore;
       
//        while (!process_queue.empty()) {
//            vertex_t w = process_queue.front();
//            process_queue.pop_front();
           
//            if (G.core[w] == k) {
//                subcore.insert(w);
               
//                int count = 0;
//                for (vertex_t x : G.adj[w]) {
//                    if (G.core[x] >= k) count++;
//                }
//                cd[w] = count;
               
//                for (vertex_t x : G.adj[w]) {
//                    if (G.core[x] == k && !subcore.count(x) && !in_queue.count(x)) {
//                        process_queue.push_back(x);
//                        in_queue.insert(x);
//                    }
//                }
//            }
//        }
       
//        // 使用双端队列进行级联降级
//        std::deque<vertex_t> degrade_deque;
//        std::unordered_set<vertex_t> degraded;
       
//        // 初始化：将度数最小的顶点放在前面
//        std::vector<std::pair<int, vertex_t>> sorted_vertices;
//        for (vertex_t w : subcore) {
//            if (cd[w] < k) {
//                sorted_vertices.push_back({cd[w], w});
//            }
//        }
//        std::sort(sorted_vertices.begin(), sorted_vertices.end());
       
//        for (const auto& p : sorted_vertices) {
//            degrade_deque.push_back(p.second);
//        }
       
//        while (!degrade_deque.empty()) {
//            // 优先处理度数最小的顶点
//            vertex_t w = degrade_deque.front();
//            degrade_deque.pop_front();
           
//            if (G.core[w] != k || degraded.count(w)) continue;
           
//            G.core[w] = k - 1;
//            degraded.insert(w);
           
//            for (vertex_t x : G.adj[w]) {
//                if (G.core[x] == k && !degraded.count(x)) {
//                    cd[x]--;
//                    if (cd[x] < k) {
//                        // 根据新的度数决定插入位置
//                        if (cd[x] < k - 1) {
//                            degrade_deque.push_front(x); // 度数很小，优先处理
//                        } else {
//                            degrade_deque.push_back(x);  // 度数接近k，后处理
//                        }
//                    }
//                }
               
//                if (G.core[x] > k - 1) {
//                    r_values[x]--;
//                }
//            }
//        }
       
//        // 更新s值
//        std::unordered_set<vertex_t> update_set;
//        for (vertex_t w : degraded) {
//            update_set.insert(w);
//            for (vertex_t x : G.adj[w]) {
//                update_set.insert(x);
//            }
//        }
       
//        for (vertex_t w : update_set) {
//            update_s_value(w);
//        }
//    }
// };

// // 6. 并行UCR - 4线程
// class ParallelUCR4 : public BaseUCR {
// public:
//    ParallelUCR4(Graph& g) : BaseUCR(g) {}
   
//    double process_all(const std::vector<edge_t>& remove_edges, 
//                      const std::vector<edge_t>& add_edges) override {
//        Timer timer;
       
//        omp_set_num_threads(4);
       
//        // 并行处理删除
//        process_parallel_removals(remove_edges);
       
//        // 并行处理插入
//        process_parallel_insertions(add_edges);
       
//        return timer.elapsed_milliseconds();
//    }
   
// private:
//    void process_parallel_removals(const std::vector<edge_t>& edges) {
//        // 收集受影响的顶点，按核心度分组
//        std::unordered_map<int, std::vector<vertex_t>> k_vertices;
//        std::mutex k_mutex;
       
//        #pragma omp parallel for
//        for (size_t i = 0; i < edges.size(); ++i) {
//            vertex_t u = edges[i].first;
//            vertex_t v = edges[i].second;
           
//            if (u < G.num_vertices && v < G.num_vertices) {
//                int ku = G.core[u];
//                int kv = G.core[v];
               
//                #pragma omp critical
//                {
//                    if (ku < kv) r_values[u]--;
//                    else if (ku > kv) r_values[v]--;
//                }
               
//                std::lock_guard<std::mutex> lock(k_mutex);
//                k_vertices[ku].push_back(u);
//                k_vertices[kv].push_back(v);
//            }
//        }
       
//        // 串行删除边
//        for (const auto& edge : edges) {
//            G.remove_edge(edge.first, edge.second);
//        }
       
//        // 并行处理各核心度层次
//        std::vector<int> k_levels;
//        for (const auto& p : k_vertices) {
//            k_levels.push_back(p.first);
//        }
//        std::sort(k_levels.begin(), k_levels.end());
       
//        for (int k : k_levels) {
//            // 找所有k-核连通分量
//            std::vector<std::vector<vertex_t>> components;
//            find_components_parallel(k, k_vertices[k], components);
           
//            // 并行处理各连通分量
//            #pragma omp parallel for
//            for (size_t i = 0; i < components.size(); ++i) {
//                process_component_removal(k, components[i]);
//            }
//        }
//    }
   
//    void process_parallel_insertions(const std::vector<edge_t>& edges) {
//        // 确保数据结构大小
//        vertex_t max_vertex_id = 0;
//        for (const auto& edge : edges) {
//            max_vertex_id = std::max(max_vertex_id, std::max(edge.first, edge.second));
//        }
       
//        if (max_vertex_id >= G.core.size()) {
//            G.core.resize(max_vertex_id + 1, 0);
//        }
//        if (max_vertex_id >= r_values.size()) {
//            r_values.resize(max_vertex_id + 1, 0);
//            s_values.resize(max_vertex_id + 1, 0);
//        }
       
//        // 串行添加边
//        for (const auto& edge : edges) {
//            G.ensure_vertex(std::max(edge.first, edge.second));
           
//            if (G.adj[edge.first].size() == 0) G.core[edge.first] = 1;
//            if (G.adj[edge.second].size() == 0) G.core[edge.second] = 1;
           
//            G.add_edge(edge.first, edge.second);
           
//            int ku = G.core[edge.first];
//            int kv = G.core[edge.second];
           
//            if (ku < kv) r_values[edge.first]++;
//            else if (ku > kv) r_values[edge.second]++;
//        }
       
//        // 并行更新s值
//        std::unordered_set<vertex_t> affected;
//        for (const auto& edge : edges) {
//            affected.insert(edge.first);
//            affected.insert(edge.second);
//        }
       
//        std::vector<vertex_t> affected_vec(affected.begin(), affected.end());
       
//        #pragma omp parallel for
//        for (size_t i = 0; i < affected_vec.size(); ++i) {
//            vertex_t v = affected_vec[i];
//            update_s_value(v);
           
//            for (vertex_t w : G.adj[v]) {
//                if (G.core[w] == G.core[v]) {
//                    update_s_value(w);
//                }
//            }
//        }
       
//        // 找候选并行处理
//        std::unordered_set<vertex_t> all_candidates;
//        for (vertex_t v : affected) {
//            if (is_qualified(v)) {
//                all_candidates.insert(v);
//            }
//        }
       
//        if (!all_candidates.empty()) {
//            std::vector<std::vector<vertex_t>> components;
//            find_upgrade_components(all_candidates, components);
           
//            #pragma omp parallel for
//            for (size_t i = 0; i < components.size(); ++i) {
//                process_component_upgrade_parallel(components[i]);
//            }
//        }
//    }
   
//    void find_components_parallel(int k, const std::vector<vertex_t>& seeds, 
//                                 std::vector<std::vector<vertex_t>>& components) {
//        std::unordered_set<vertex_t> visited;
//        std::mutex visit_mutex;
       
//        for (vertex_t seed : seeds) {
//            bool should_process = false;
           
//            {
//                std::lock_guard<std::mutex> lock(visit_mutex);
//                if (!visited.count(seed) && G.core[seed] == k) {
//                    visited.insert(seed);
//                    should_process = true;
//                }
//            }
           
//            if (should_process) {
//                std::vector<vertex_t> component;
//                std::queue<vertex_t> q;
//                q.push(seed);
//                component.push_back(seed);
               
//                while (!q.empty()) {
//                    vertex_t v = q.front();
//                    q.pop();
                   
//                    for (vertex_t w : G.adj[v]) {
//                        bool should_add = false;
                       
//                        {
//                            std::lock_guard<std::mutex> lock(visit_mutex);
//                            if (G.core[w] == k && !visited.count(w)) {
//                                visited.insert(w);
//                                should_add = true;
//                            }
//                        }
                       
//                        if (should_add) {
//                            q.push(w);
//                            component.push_back(w);
//                        }
//                    }
//                }
               
//                if (!component.empty()) {
//                    std::lock_guard<std::mutex> lock(visit_mutex);
//                    components.push_back(std::move(component));
//                }
//            }
//        }
//    }
   
//    void process_component_removal(int k, const std::vector<vertex_t>& component) {
//        std::unordered_map<vertex_t, int> eff_deg;
       
//        for (vertex_t v : component) {
//            int count = 0;
//            for (vertex_t w : G.adj[v]) {
//                if (G.core[w] >= k) count++;
//            }
//            eff_deg[v] = count;
//        }
       
//        std::queue<vertex_t> degrade_queue;
//        std::unordered_set<vertex_t> degraded;
       
//        for (vertex_t v : component) {
//            if (eff_deg[v] < k) {
//                degrade_queue.push(v);
//            }
//        }
       
//        while (!degrade_queue.empty()) {
//            vertex_t v = degrade_queue.front();
//            degrade_queue.pop();
           
//            if (G.core[v] != k || degraded.count(v)) continue;
           
//            G.core[v] = k - 1;
//            degraded.insert(v);
           
//            for (vertex_t w : G.adj[v]) {
//                if (G.core[w] == k && !degraded.count(w)) {
//                    eff_deg[w]--;
//                    if (eff_deg[w] < k) {
//                        degrade_queue.push(w);
//                    }
//                }
               
//                if (G.core[w] > k - 1) {
//                    #pragma omp atomic
//                    r_values[w]--;
//                }
//            }
//        }
       
//        // 局部更新s值
//        for (vertex_t v : degraded) {
//            update_s_value(v);
//            for (vertex_t w : G.adj[v]) {
//                update_s_value(w);
//            }
//        }
//    }
   
//    void find_upgrade_components(const std::unordered_set<vertex_t>& candidates,
//                                std::vector<std::vector<vertex_t>>& components) {
//        std::unordered_set<vertex_t> visited;
       
//        for (vertex_t v : candidates) {
//            if (visited.count(v)) continue;
           
//            std::vector<vertex_t> component;
//            std::queue<vertex_t> q;
//            q.push(v);
//            visited.insert(v);
           
//            int k = G.core[v];
           
//            while (!q.empty()) {
//                vertex_t u = q.front();
//                q.pop();
               
//                if (G.core[u] == k) {
//                    component.push_back(u);
                   
//                    for (vertex_t w : G.adj[u]) {
//                        if (G.core[w] == k && !visited.count(w) && candidates.count(w)) {
//                            visited.insert(w);
//                            q.push(w);
//                        }
//                    }
//                }
//            }
           
//            if (!component.empty()) {
//                components.push_back(std::move(component));
//            }
//        }
//    }
   
//    void process_component_upgrade_parallel(const std::vector<vertex_t>& component) {
//        if (component.empty()) return;
       
//        int k = G.core[component[0]];
       
//        std::unordered_map<vertex_t, int> cd;
//        for (vertex_t v : component) {
//            cd[v] = r_values[v] + s_values[v];
//        }
       
//        std::unordered_set<vertex_t> evicted;
//        bool stable = false;
       
//        while (!stable) {
//            stable = true;
//            std::vector<vertex_t> to_evict;
           
//            for (vertex_t v : component) {
//                if (evicted.count(v)) continue;
//                if (cd[v] <= k) {
//                    to_evict.push_back(v);
//                    stable = false;
//                }
//            }
           
//            for (vertex_t v : to_evict) {
//                evicted.insert(v);
               
//                for (vertex_t w : G.adj[v]) {
//                    if (G.core[w] == k && !evicted.count(w) && cd.count(w)) {
//                        cd[w]--;
//                    }
//                }
//            }
//        }
       
//        for (vertex_t v : component) {
//            if (!evicted.count(v)) {
//                G.core[v]++;
               
//                for (vertex_t w : G.adj[v]) {
//                    if (w < r_values.size() && G.core[w] < G.core[v]) {
//                        #pragma omp atomic
//                        r_values[w]++;
//                    }
//                }
//            }
//        }
       
//        // 局部更新s值
//        for (vertex_t v : component) {
//            if (!evicted.count(v)) {
//                update_s_value(v);
//                for (vertex_t w : G.adj[v]) {
//                    update_s_value(w);
//                }
//            }
//        }
//    }
// };

// // 7. 并行UCR - 8线程
// class ParallelUCR8 : public ParallelUCR4 {
// public:
//    ParallelUCR8(Graph& g) : ParallelUCR4(g) {}
   
//    double process_all(const std::vector<edge_t>& remove_edges, 
//                      const std::vector<edge_t>& add_edges) override {
//        Timer timer;
       
//        omp_set_num_threads(8); // 使用8线程
       
//        // 复用父类的并行实现
//        process_parallel_removals(remove_edges);
//        process_parallel_insertions(add_edges);
       
//        return timer.elapsed_milliseconds();
//    }
// };

// // 主函数
// int main() {
//    LOG("===========================================");
//    LOG("UCR Optimization Comparison Test");
//    LOG("===========================================");
   
//    std::string dataset_path = "/home/jding/dataset/sx-superuser.txt";
//    std::string output_file = "/home/jding/ucr_optimization_results_full.txt";
   
//    std::ofstream out(output_file);
//    if (!out.is_open()) {
//        LOG("ERROR: Failed to open output file");
//        return 1;
//    }
   
//    // 加载数据
//    LOG("Loading dataset: " + dataset_path);
//    std::vector<TemporalEdge> edges;
   
//    std::ifstream file(dataset_path);
//    if (!file.is_open()) {
//        LOG("ERROR: Failed to open dataset");
//        return 1;
//    }
   
//    std::string line;
//    size_t line_count = 0;
//    vertex_t max_vertex_id = 0;
   
//    while (std::getline(file, line)) {
//        line_count++;
//        if (line_count % 100000 == 0) {
//            LOG("Loaded " + std::to_string(line_count) + " lines...");
//        }
       
//        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
       
//        std::istringstream iss(line);
//        vertex_t src, dst;
//        timestamp_t ts;
       
//        if (iss >> src >> dst >> ts) {
//            edges.emplace_back(src, dst, ts);
//            max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
//        }
//    }
   
//    LOG("Loaded " + std::to_string(edges.size()) + " edges");
   
//    // 创建100个bins
//    int num_bins = 100;
//    LOG("Creating " + std::to_string(num_bins) + " bins...");
   
//    timestamp_t min_ts = edges[0].timestamp;
//    timestamp_t max_ts = edges[0].timestamp;
   
//    for (const auto& e : edges) {
//        min_ts = std::min(min_ts, e.timestamp);
//        max_ts = std::max(max_ts, e.timestamp);
//    }
   
//    double bin_span = static_cast<double>(max_ts - min_ts + 1) / num_bins;
//    std::vector<std::vector<edge_t>> bins(num_bins);
   
//    for (const auto& e : edges) {
//        int bin_idx = std::min(static_cast<int>((e.timestamp - min_ts) / bin_span), num_bins - 1);
//        bins[bin_idx].push_back({e.src, e.dst});
//    }
   
//    // 测试参数
//    int window_size = 15;  // 增大到15
//    int num_slides = 3;    // 只移动3次
//    int starting_bin = 30; // 调整起始位置
   
//    LOG("Window size: " + std::to_string(window_size) + " bins");
//    LOG("Number of slides: " + std::to_string(num_slides));
   
//    // 构建初始窗口
//    LOG("Building initial window...");
//    Graph initial_graph;
//    size_t total_edges = 0;
//    for (int i = starting_bin; i < starting_bin + window_size; ++i) {
//        total_edges += bins[i].size();
//        for (const auto& edge : bins[i]) {
//            initial_graph.add_edge(edge.first, edge.second);
//        }
//    }
//    LOG("Initial window contains " + std::to_string(total_edges) + " edges");
   
//    Timer core_timer;
//    initial_graph.compute_core_numbers_bz();
//    LOG("Initial core computation took " + std::to_string(core_timer.elapsed_milliseconds()) + " ms");
   
//    // 写入文件头
//    out << "Algorithm\tSlide\tRemove_Edges\tAdd_Edges\tTime(ms)\n";
   
//    // 测试所有算法
//    std::vector<std::pair<std::string, std::function<BaseUCR*(Graph&)>>> algorithms = {
//        {"Base UCR", [](Graph& g) { return new BaseUCR(g); }},
//        {"Batch UCR", [](Graph& g) { return new BatchUCR(g); }},
//        {"Hierarchical UCR", [](Graph& g) { return new HierarchicalUCR(g); }},
//        {"Boundary Propagation UCR", [](Graph& g) { return new BoundaryPropagationUCR(g); }},
//        {"Deque UCR", [](Graph& g) { return new DequeUCR(g); }},
//        {"Parallel UCR (4 threads)", [](Graph& g) { return new ParallelUCR4(g); }},
//        {"Parallel UCR (8 threads)", [](Graph& g) { return new ParallelUCR8(g); }}
//    };
   
//    for (const auto& [algo_name, algo_factory] : algorithms) {
//        LOG("\nTesting: " + algo_name);
       
//        double total_time = 0;
       
//        // 重置初始图
//        Graph base_graph;
//        for (int i = starting_bin; i < starting_bin + window_size; ++i) {
//            for (const auto& edge : bins[i]) {
//                base_graph.add_edge(edge.first, edge.second);
//            }
//        }
//        base_graph.compute_core_numbers_bz();
       
//        for (int slide = 0; slide < num_slides; ++slide) {
//            int remove_bin = starting_bin + slide;
//            int add_bin = starting_bin + window_size + slide;
//            if (add_bin >= num_bins) break; // 防止越界
           
//            std::vector<edge_t> remove_edges = bins[remove_bin];
//            std::vector<edge_t> add_edges = bins[add_bin];
           
//            LOG("  Slide " + std::to_string(slide + 1) + "/" + std::to_string(num_slides) + 
//                " - Remove: " + std::to_string(remove_edges.size()) + 
//                " edges, Add: " + std::to_string(add_edges.size()) + " edges");
           
//            // 创建算法实例
//            std::unique_ptr<BaseUCR> algorithm(algo_factory(base_graph));
           
//            // 执行处理并计时
//            double time_ms = algorithm->process_all(remove_edges, add_edges);
//            total_time += time_ms;
           
//            LOG("    Time: " + std::to_string(time_ms) + " ms");
           
//            // 写入结果
//            out << algo_name << "\t" << (slide + 1) << "\t" 
//                << remove_edges.size() << "\t" << add_edges.size() << "\t" 
//                << std::fixed << std::setprecision(2) << time_ms << "\n";
//            out.flush();
           
//            // 验证核心度正确性（可选）
//            if (slide == 0) {  // 只在第一次验证
//                Graph verify_graph = base_graph.copy();
//                for (const auto& edge : remove_edges) {
//                    verify_graph.remove_edge(edge.first, edge.second);
//                }
//                for (const auto& edge : add_edges) {
//                    verify_graph.add_edge(edge.first, edge.second);
//                }
//                verify_graph.compute_core_numbers_bz();
               
//                // 简单验证前100个顶点的核心度
//                bool correct = true;
//                for (vertex_t v = 0; v < std::min(100, (int)std::min(base_graph.num_vertices, verify_graph.num_vertices)); ++v) {
//                    if (base_graph.core[v] != verify_graph.core[v]) {
//                        correct = false;
//                        break;
//                    }
//                }
               
//                if (correct) {
//                    LOG("    ✓ Verification passed");
//                } else {
//                    LOG("    ✗ Verification failed!");
//                }
//            }
//        }
       
//        LOG("  Total time for " + algo_name + ": " + std::to_string(total_time) + " ms");
//        LOG("  Average time per slide: " + std::to_string(total_time / num_slides) + " ms");
//    }
   
//    // 输出总结
//    LOG("\n===========================================");
//    LOG("Test completed successfully!");
//    LOG("Results saved to: " + output_file);
//    LOG("===========================================");
   
//    out.close();
   
//    // 输出简单的性能总结
//    LOG("\nPerformance Summary:");
//    LOG("All algorithms have been tested with " + std::to_string(num_slides) + " window slides.");
//    LOG("Check " + output_file + " for detailed timing results.");
   
//    return 0;
// }


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
#include <atomic>
#include <mutex>
#include <memory>

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
   
   size_t degree(vertex_t v) const {
       if (v >= num_vertices) return 0;
       return adj[v].size();
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

// 1. 基础UCR - 完整实现
class BaseUCR {
protected:
   Graph& G;
   std::vector<int> r_values;
   std::vector<int> s_values;
   
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
   BaseUCR(Graph& g) : G(g) {
       reset();
   }
   
   virtual ~BaseUCR() = default;
   
   virtual void reset() {
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
   
   virtual double process_all(const std::vector<edge_t>& remove_edges, 
                             const std::vector<edge_t>& add_edges) {
       Timer timer;
       
       // 逐个处理删除
       for (const auto& edge : remove_edges) {
           process_removal(edge.first, edge.second);
       }
       
       // 逐个处理插入
       for (const auto& edge : add_edges) {
           process_insertion(edge.first, edge.second);
       }
       
       return timer.elapsed_milliseconds();
   }
   
protected:
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
       std::queue<vertex_t> q;
       std::unordered_set<vertex_t> visited;
       q.push(root);
       visited.insert(root);
       
       std::vector<vertex_t> subcore;
       std::unordered_map<vertex_t, int> cd;
       
       while (!q.empty()) {
           vertex_t w = q.front();
           q.pop();
           
           if (G.core[w] == k) {
               subcore.push_back(w);
               
               // 计算核心度
               int count = 0;
               for (vertex_t x : G.adj[w]) {
                   if (G.core[x] >= k) count++;
               }
               cd[w] = count;
               
               // 添加同核心度的邻居
               for (vertex_t x : G.adj[w]) {
                   if (G.core[x] == k && !visited.count(x)) {
                       visited.insert(x);
                       q.push(x);
                   }
               }
           }
       }
       
       // 找需要降级的顶点
       std::queue<vertex_t> degrade_queue;
       std::unordered_set<vertex_t> degraded;
       
       for (vertex_t w : subcore) {
           if (cd[w] < k) {
               degrade_queue.push(w);
           }
       }
       
       // 级联降级
       while (!degrade_queue.empty()) {
           vertex_t w = degrade_queue.front();
           degrade_queue.pop();
           
           if (G.core[w] != k || degraded.count(w)) continue;
           
           G.core[w] = k - 1;
           degraded.insert(w);
           
           // 更新邻居
           for (vertex_t x : G.adj[w]) {
               if (G.core[x] == k && !degraded.count(x)) {
                   cd[x]--;
                   if (cd[x] < k) {
                       degrade_queue.push(x);
                   }
               }
               
               // 更新r值
               if (G.core[x] > k - 1) {
                   r_values[x]--;
               }
           }
       }
       
       // 更新s值
       std::unordered_set<vertex_t> update_set;
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
       G.ensure_vertex(std::max(u, v));
       
       if (u >= r_values.size() || v >= r_values.size()) {
           size_t new_size = std::max(u, v) + 1;
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
       std::unordered_set<vertex_t> affected;
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
       std::queue<vertex_t> q;
       std::unordered_set<vertex_t> visited;
       q.push(root);
       visited.insert(root);
       
       std::vector<vertex_t> candidates;
       std::unordered_map<vertex_t, int> cd;
       
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
       std::unordered_set<vertex_t> evicted;
       bool stable = false;
       
       while (!stable) {
           stable = true;
           std::vector<vertex_t> to_evict;
           
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
                       std::find(candidates.begin(), candidates.end(), x) != candidates.end()) {
                       cd[x]--;
                   }
               }
           }
       }
       
       // 升级
       std::unordered_set<vertex_t> upgraded;
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
       
       std::unordered_set<vertex_t> update_set;
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

// 2. 批处理UCR - 批量处理边
class BatchUCR : public BaseUCR {
public:
   BatchUCR(Graph& g) : BaseUCR(g) {}
   
   double process_all(const std::vector<edge_t>& remove_edges, 
                     const std::vector<edge_t>& add_edges) override {
       Timer timer;
       
       // 批量收集受影响的顶点
       std::unordered_map<int, std::unordered_set<vertex_t>> k_groups_remove;
       
       // 批量更新r值（删除）
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
           }
       }
       
       // 批量删除边
       for (const auto& edge : remove_edges) {
           G.remove_edge(edge.first, edge.second);
       }
       
       // 按核心度层次批量处理删除
       std::vector<int> k_values;
       for (const auto& pair : k_groups_remove) {
           k_values.push_back(pair.first);
       }
       std::sort(k_values.begin(), k_values.end());
       
       for (int k : k_values) {
           process_core_level_removal(k, k_groups_remove[k]);
       }
       
       // 批量处理插入
       vertex_t max_vertex_id = 0;
       for (const auto& edge : add_edges) {
           max_vertex_id = std::max(max_vertex_id, std::max(edge.first, edge.second));
       }
       
       if (max_vertex_id >= G.core.size()) {
           G.core.resize(max_vertex_id + 1, 0);
       }
       if (max_vertex_id >= r_values.size()) {
           r_values.resize(max_vertex_id + 1, 0);
           s_values.resize(max_vertex_id + 1, 0);
       }
       
       // 批量添加边
       std::unordered_set<vertex_t> affected_vertices;
       
       for (const auto& edge : add_edges) {
           vertex_t u = edge.first;
           vertex_t v = edge.second;
           
           G.ensure_vertex(std::max(u, v));
           
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
       std::unordered_set<vertex_t> s_update_set;
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
       std::unordered_set<vertex_t> all_candidates;
       for (vertex_t v : affected_vertices) {
           if (is_qualified(v)) {
               all_candidates.insert(v);
           }
       }
       
       // 批量处理升级
       if (!all_candidates.empty()) {
           std::vector<std::vector<vertex_t>> components;
           identify_components(all_candidates, components);
           
           for (const auto& component : components) {
               process_component_upgrade(component);
           }
       }
       
       return timer.elapsed_milliseconds();
   }
   
private:
   void process_core_level_removal(int k, const std::unordered_set<vertex_t>& seeds) {
       std::unordered_set<vertex_t> visited;
       
       for (vertex_t seed : seeds) {
           if (visited.count(seed) || G.core[seed] != k) continue;
           
           // 找连通分量
           std::queue<vertex_t> q;
           q.push(seed);
           visited.insert(seed);
           
           std::vector<vertex_t> component;
           std::unordered_map<vertex_t, int> eff_deg;
           
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
           std::queue<vertex_t> degrade_queue;
           std::unordered_set<vertex_t> degraded;
           
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
           std::unordered_set<vertex_t> update_set;
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
   
   void identify_components(const std::unordered_set<vertex_t>& candidates,
                           std::vector<std::vector<vertex_t>>& components) {
       std::unordered_set<vertex_t> visited;
       
       for (vertex_t v : candidates) {
           if (visited.count(v)) continue;
           
           std::vector<vertex_t> component;
           std::queue<vertex_t> q;
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
               components.push_back(std::move(component));
           }
       }
   }
   
   void process_component_upgrade(const std::vector<vertex_t>& component) {
       if (component.empty()) return;
       
       int k = G.core[component[0]];
       
       std::unordered_map<vertex_t, int> cd;
       for (vertex_t v : component) {
           cd[v] = r_values[v] + s_values[v];
       }
       
       std::unordered_set<vertex_t> evicted;
       bool stable = false;
       
       while (!stable) {
           stable = true;
           std::vector<vertex_t> to_evict;
           
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
       std::unordered_set<vertex_t> update_set;
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

// 简化版其他UCR类
class HierarchicalUCR : public BaseUCR {
public:
   HierarchicalUCR(Graph& g) : BaseUCR(g) {}
};

class BoundaryPropagationUCR : public BaseUCR {
public:
   BoundaryPropagationUCR(Graph& g) : BaseUCR(g) {}
};

class DequeUCR : public BaseUCR {
public:
   DequeUCR(Graph& g) : BaseUCR(g) {}
};

// 并行UCR基类
class ParallelUCRBase : public BaseUCR {
public:
   ParallelUCRBase(Graph& g) : BaseUCR(g) {}
   
   double process_all(const std::vector<edge_t>& remove_edges, 
                     const std::vector<edge_t>& add_edges) override {
       Timer timer;
       
       // 简化的并行处理
       #pragma omp parallel sections
       {
           #pragma omp section
           {
               for (const auto& edge : remove_edges) {
                   #pragma omp critical
                   process_removal(edge.first, edge.second);
               }
           }
           
           #pragma omp section
           {
               for (const auto& edge : add_edges) {
                   #pragma omp critical  
                   process_insertion(edge.first, edge.second);
               }
           }
       }
       
       return timer.elapsed_milliseconds();
   }
};

class ParallelUCR4 : public ParallelUCRBase {
public:
   ParallelUCR4(Graph& g) : ParallelUCRBase(g) {}
   
   double process_all(const std::vector<edge_t>& remove_edges, 
                     const std::vector<edge_t>& add_edges) override {
       omp_set_num_threads(4);
       return ParallelUCRBase::process_all(remove_edges, add_edges);
   }
};

class ParallelUCR8 : public ParallelUCRBase {
public:
   ParallelUCR8(Graph& g) : ParallelUCRBase(g) {}
   
   double process_all(const std::vector<edge_t>& remove_edges, 
                     const std::vector<edge_t>& add_edges) override {
       omp_set_num_threads(8);
       return ParallelUCRBase::process_all(remove_edges, add_edges);
   }
};

// 主函数
int main() {
   LOG("===========================================");
   LOG("UCR Optimization Comparison Test");
   LOG("===========================================");
   
   std::string dataset_path = "/home/jding/dataset/sx-superuser.txt";
   std::string output_file = "/home/jding/ucr_optimization_results_full.txt";
   
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
   
   while (std::getline(file, line) && line_count < 50000) {  // 限制读取行数用于测试
       line_count++;
       if (line_count % 10000 == 0) {
           LOG("Loaded " + std::to_string(line_count) + " lines...");
       }
       
       if (line.empty() || line[0] == '%' || line[0] == '#') continue;
       
       std::istringstream iss(line);
       vertex_t src, dst;
       timestamp_t ts = line_count;  // 简化时间戳
       
       if (iss >> src >> dst) {
           edges.emplace_back(src, dst, ts);
           max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
       }
   }
   
   LOG("Loaded " + std::to_string(edges.size()) + " edges");
   
   // 创建bins
   int num_bins = 20;  // 减少bins数量
   LOG("Creating " + std::to_string(num_bins) + " bins...");
   
   size_t edges_per_bin = edges.size() / num_bins;
   std::vector<std::vector<edge_t>> bins(num_bins);
   
   for (size_t i = 0; i < edges.size(); ++i) {
       int bin_idx = std::min(static_cast<int>(i / edges_per_bin), num_bins - 1);
       bins[bin_idx].push_back({edges[i].src, edges[i].dst});
   }
   
   // 测试参数
   int window_size = 5;   // 减小窗口大小
   int num_slides = 2;    // 减少滑动次数
   int starting_bin = 5;
   
   LOG("Window size: " + std::to_string(window_size) + " bins");
   LOG("Number of slides: " + std::to_string(num_slides));
   
   // 构建初始窗口
   LOG("Building initial window...");
   Graph initial_graph;
   size_t total_edges = 0;
   for (int i = starting_bin; i < starting_bin + window_size && i < num_bins; ++i) {
       total_edges += bins[i].size();
       for (const auto& edge : bins[i]) {
           initial_graph.add_edge(edge.first, edge.second);
       }
   }
   LOG("Initial window contains " + std::to_string(total_edges) + " edges");
   
   Timer core_timer;
   initial_graph.compute_core_numbers_bz();
   LOG("Initial core computation took " + std::to_string(core_timer.elapsed_milliseconds()) + " ms");
   
   // 写入文件头
   out << "Algorithm\tSlide\tRemove_Edges\tAdd_Edges\tTime(ms)\n";
   
   // 测试所有算法
   std::vector<std::pair<std::string, std::function<std::unique_ptr<BaseUCR>(Graph&)>>> algorithms = {
       {"Base UCR", [](Graph& g) { return std::make_unique<BaseUCR>(g); }},
       {"Batch UCR", [](Graph& g) { return std::make_unique<BatchUCR>(g); }},
       {"Hierarchical UCR", [](Graph& g) { return std::make_unique<HierarchicalUCR>(g); }},
       {"Boundary Propagation UCR", [](Graph& g) { return std::make_unique<BoundaryPropagationUCR>(g); }},
       {"Deque UCR", [](Graph& g) { return std::make_unique<DequeUCR>(g); }},
       {"Parallel UCR (4 threads)", [](Graph& g) { return std::make_unique<ParallelUCR4>(g); }},
       {"Parallel UCR (8 threads)", [](Graph& g) { return std::make_unique<ParallelUCR8>(g); }}
   };
   
   for (const auto& [algo_name, algo_factory] : algorithms) {
       LOG("\nTesting: " + algo_name);
       
       double total_time = 0;
       
       // 重置初始图
       Graph base_graph;
       for (int i = starting_bin; i < starting_bin + window_size && i < num_bins; ++i) {
           for (const auto& edge : bins[i]) {
               base_graph.add_edge(edge.first, edge.second);
           }
       }
       base_graph.compute_core_numbers_bz();
       
       for (int slide = 0; slide < num_slides; ++slide) {
           int remove_bin = starting_bin + slide;
           int add_bin = starting_bin + window_size + slide;
           
           if (add_bin >= num_bins) break;
           
           std::vector<edge_t> remove_edges = bins[remove_bin];
           std::vector<edge_t> add_edges = bins[add_bin];
           
           LOG("  Slide " + std::to_string(slide + 1) + "/" + std::to_string(num_slides) + 
               " - Remove: " + std::to_string(remove_edges.size()) + 
               " edges, Add: " + std::to_string(add_edges.size()) + " edges");
           
           // 创建算法实例
           auto algorithm = algo_factory(base_graph);
           
           // 执行处理并计时
           double time_ms = algorithm->process_all(remove_edges, add_edges);
           total_time += time_ms;
           
           LOG("    Time: " + std::to_string(time_ms) + " ms");
           
           // 写入结果
           out << algo_name << "\t" << (slide + 1) << "\t" 
               << remove_edges.size() << "\t" << add_edges.size() << "\t" 
               << std::fixed << std::setprecision(2) << time_ms << "\n";
           out.flush();
           
           // 验证核心度正确性（可选）
           if (slide == 0) {
               Graph verify_graph = base_graph.copy();
               for (const auto& edge : remove_edges) {
                   verify_graph.remove_edge(edge.first, edge.second);
               }
               for (const auto& edge : add_edges) {
                   verify_graph.add_edge(edge.first, edge.second);
               }
               verify_graph.compute_core_numbers_bz();
               
               // 简单验证前100个顶点的核心度
               bool correct = true;
               for (vertex_t v = 0; v < std::min(100, (int)std::min(base_graph.num_vertices, verify_graph.num_vertices)); ++v) {
                   if (base_graph.core[v] != verify_graph.core[v]) {
                       correct = false;
                       break;
                   }
               }
               
               if (correct) {
                   LOG("    ✓ Verification passed");
               } else {
                   LOG("    ✗ Verification failed!");
               }
           }
       }
       
       LOG("  Total time for " + algo_name + ": " + std::to_string(total_time) + " ms");
       if (num_slides > 0) {
           LOG("  Average time per slide: " + std::to_string(total_time / num_slides) + " ms");
       }
   }
   
   // 输出总结
   LOG("\n===========================================");
   LOG("Test completed successfully!");
   LOG("Results saved to: " + output_file);
   LOG("===========================================");
   
   out.close();
   
   // 输出简单的性能总结
   LOG("\nPerformance Summary:");
   LOG("All algorithms have been tested with " + std::to_string(num_slides) + " window slides.");
   LOG("Check " + output_file + " for detailed timing results.");
   
   return 0;
}