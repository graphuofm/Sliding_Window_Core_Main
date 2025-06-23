#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <map>
#include <atomic>
#include <bitset>

// 定义顶点和边的类型
using vertex_t = int;
using edge_t = std::pair<vertex_t, vertex_t>;
using timestamp_t = long long;

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
    
    double elapsed_microseconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end_time - start_time).count();
    }
    
    double elapsed_milliseconds() const {
        return elapsed_microseconds() / 1000.0;
    }
    
    double elapsed_seconds() const {
        return elapsed_milliseconds() / 1000.0;
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
};

// 优化的邻接表结构
class FastAdjacencyList {
private:
    std::vector<std::vector<vertex_t>> adj_lists;
    
public:
    FastAdjacencyList() {}
    
    void resize(size_t size) {
        if (size > adj_lists.size()) {
            adj_lists.resize(size);
        }
    }
    
    size_t size() const {
        return adj_lists.size();
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        size_t max_idx = std::max(u, v);
        if (max_idx >= adj_lists.size()) {
            resize(max_idx + 1);
        }
        
        // 检查边是否已存在，以避免重复
        if (std::find(adj_lists[u].begin(), adj_lists[u].end(), v) == adj_lists[u].end()) {
            adj_lists[u].push_back(v);
        }
        if (std::find(adj_lists[v].begin(), adj_lists[v].end(), u) == adj_lists[v].end()) {
            adj_lists[v].push_back(u);
        }
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        if (u >= adj_lists.size() || v >= adj_lists.size()) return;
        
        auto it_u = std::find(adj_lists[u].begin(), adj_lists[u].end(), v);
        if (it_u != adj_lists[u].end()) {
            *it_u = adj_lists[u].back();
            adj_lists[u].pop_back();
        }
        
        auto it_v = std::find(adj_lists[v].begin(), adj_lists[v].end(), u);
        if (it_v != adj_lists[v].end()) {
            *it_v = adj_lists[v].back();
            adj_lists[v].pop_back();
        }
    }
    
    const std::vector<vertex_t>& neighbors(vertex_t v) const {
        static const std::vector<vertex_t> empty;
        if (v >= adj_lists.size()) return empty;
        return adj_lists[v];
    }
    
    size_t degree(vertex_t v) const {
        if (v >= adj_lists.size()) return 0;
        return adj_lists[v].size();
    }
    
    void clear() {
        for (auto& list : adj_lists) {
            list.clear();
        }
    }
    
    size_t memory_usage() const {
        size_t total = 0;
        for (const auto& list : adj_lists) {
            total += list.capacity() * sizeof(vertex_t);
        }
        return total;
    }
};

// 优化的图结构
class Graph {
public:
    FastAdjacencyList adj;  // 邻接表
    std::vector<int> core;  // 核心度值
    
    Graph() {}
    
    size_t n() const {
        return adj.size();
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        size_t max_idx = std::max(u, v);
        if (max_idx >= core.size()) {
            core.resize(max_idx + 1, 0);
        }
        adj.add_edge(u, v);
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        adj.remove_edge(u, v);
    }
    
    void ensure_vertex(vertex_t v) {
        if (v >= core.size()) {
            core.resize(v + 1, 0);
        }
        if (v >= adj.size()) {
            adj.resize(v + 1);
        }
    }
    
    void compute_core_numbers() {
        size_t n_vertices = adj.size();
        std::fill(core.begin(), core.end(), 0);
        
        if (n_vertices == 0) return;
        
        // 计算最大度数
        int max_degree = 0;
        std::vector<int> degree(n_vertices);
        
        for (vertex_t v = 0; v < n_vertices; ++v) {
            degree[v] = adj.degree(v);
            max_degree = std::max(max_degree, degree[v]);
        }
        
        // 按度数分桶
        std::vector<std::vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < n_vertices; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        // 核心度算法
        std::vector<bool> processed(n_vertices, false);
        int current_core = 0;
        
        for (int d = 0; d <= max_degree; ++d) {
            for (vertex_t v : bins[d]) {
                if (processed[v]) continue;
                
                current_core = d;
                core[v] = current_core;
                processed[v] = true;
                
                for (vertex_t w : adj.neighbors(v)) {
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
    
    size_t memory_usage() const {
        return core.capacity() * sizeof(int) + adj.memory_usage();
    }
};

// 优化的UCR实现
class OptimizedUCRCore {
protected:
    struct CompactUCR {
        uint16_t r;  // 高核心度邻居数量
        uint16_t s;  // 有资格的同核心度邻居数量
        
        CompactUCR() : r(0), s(0) {}
        CompactUCR(uint16_t r_val, uint16_t s_val) : r(r_val), s(s_val) {}
    };
    
    Graph& G;
    std::vector<CompactUCR> ucr_values;
    std::vector<bool> is_candidate;
    
    // 用于快速存储和检索核心度为特定值的顶点
    std::vector<std::vector<vertex_t>> core_level_vertices;
    int max_core_level;
    
public:
    OptimizedUCRCore(Graph& g) : G(g), max_core_level(0) {
        reset(true);
    }
    
    void reset(bool full_reset = false) {
        size_t n = G.n();
        
        if (ucr_values.size() < n) {
            ucr_values.resize(n, CompactUCR(0, 0));
        }
        
        if (is_candidate.size() < n) {
            is_candidate.resize(n, false);
        }
        
        if (full_reset) {
            // 计算最大核心度并构建核心度层次索引
            max_core_level = 0;
            for (vertex_t v = 0; v < n; ++v) {
                max_core_level = std::max(max_core_level, G.core[v]);
            }
            
            core_level_vertices.clear();
            core_level_vertices.resize(max_core_level + 1);
            
            for (vertex_t v = 0; v < n; ++v) {
                int k = G.core[v];
                if (k <= max_core_level) {
                    core_level_vertices[k].push_back(v);
                }
            }
            
            // 计算r值
            #pragma omp parallel for schedule(dynamic, 1000)
            for (vertex_t v = 0; v < n; ++v) {
                int k = G.core[v];
                int r_count = 0;
                
                for (vertex_t w : G.adj.neighbors(v)) {
                    if (G.core[w] > k) {
                        ++r_count;
                    }
                }
                
                ucr_values[v].r = r_count;
                ucr_values[v].s = 0; // 初始化s值为0
            }
            
            // 计算s值 - 按核心度层次从高到低
            for (int k = max_core_level; k >= 0; --k) {
                // 对于每个核心度层，计算该层中顶点的s值
                std::vector<vertex_t>& vertices = core_level_vertices[k];
                
                // 首次计算每个顶点的s值
                #pragma omp parallel for schedule(dynamic, 1000)
                for (size_t i = 0; i < vertices.size(); ++i) {
                    vertex_t v = vertices[i];
                    update_s_value(v);
                }
                
                // 迭代求解s值直至稳定
                bool stable = false;
                while (!stable) {
                    stable = true;
                    
                    std::vector<int> new_s_values(vertices.size(), 0);
                    
                    #pragma omp parallel for schedule(dynamic, 1000)
                    for (size_t i = 0; i < vertices.size(); ++i) {
                        vertex_t v = vertices[i];
                        int old_s = ucr_values[v].s;
                        new_s_values[i] = calculate_s_value(v);
                        
                        if (old_s != new_s_values[i]) {
                            #pragma omp atomic write
                            stable = false;
                        }
                    }
                    
                    if (!stable) {
                        #pragma omp parallel for schedule(dynamic, 1000)
                        for (size_t i = 0; i < vertices.size(); ++i) {
                            ucr_values[vertices[i]].s = new_s_values[i];
                        }
                    }
                }
            }
        }
    }
    
    void update_s_value(vertex_t v) {
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj.neighbors(v)) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        ucr_values[v].s = s_count;
    }
    
    int calculate_s_value(vertex_t v) {
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj.neighbors(v)) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        return s_count;
    }
    
    bool is_qualified(vertex_t v) const {
        int k = G.core[v];
        return ucr_values[v].r + ucr_values[v].s > k;
    }
    
    bool can_upgrade(vertex_t v) const {
        return ucr_values[v].r + ucr_values[v].s > G.core[v];
    }
    
    void fast_promote_core_1_vertices(std::unordered_set<vertex_t>& candidates, 
                                      std::unordered_set<vertex_t>& promoted) {
        std::vector<vertex_t> core_1_candidates;
        
        // 收集所有核心度为1的候选顶点
        for (vertex_t v : candidates) {
            if (G.core[v] == 1) {
                core_1_candidates.push_back(v);
            }
        }
        
        if (core_1_candidates.empty()) return;
        
        // 标记所有潜在的可提升顶点
        std::vector<bool> potential_promotion(core_1_candidates.size(), false);
        
        // 第一步：识别有至少两个非零核心度邻居的顶点
        #pragma omp parallel for schedule(dynamic, 1000)
        for (size_t i = 0; i < core_1_candidates.size(); ++i) {
            vertex_t v = core_1_candidates[i];
            
            int valid_neighbors = 0;
            for (vertex_t w : G.adj.neighbors(v)) {
                if (G.core[w] >= 1) {
                    valid_neighbors++;
                    if (valid_neighbors >= 2) {
                        potential_promotion[i] = true;
                        break;
                    }
                }
            }
        }
        
        // 构建所有核心度为1且有潜力的顶点的诱导子图
        std::unordered_map<vertex_t, size_t> vertex_to_index;
        std::vector<vertex_t> index_to_vertex;
        
        for (size_t i = 0; i < core_1_candidates.size(); ++i) {
            if (potential_promotion[i]) {
                vertex_t v = core_1_candidates[i];
                vertex_to_index[v] = index_to_vertex.size();
                index_to_vertex.push_back(v);
            }
        }
        
        // 如果没有潜在可提升顶点，直接返回
        if (index_to_vertex.empty()) return;
        
        // 构建邻接矩阵表示诱导子图
        std::vector<std::vector<bool>> adj_matrix(index_to_vertex.size(), 
                                                 std::vector<bool>(index_to_vertex.size(), false));
        
        // 填充邻接矩阵
        for (size_t i = 0; i < index_to_vertex.size(); ++i) {
            vertex_t v = index_to_vertex[i];
            
            for (vertex_t w : G.adj.neighbors(v)) {
                if (G.core[w] == 1 && vertex_to_index.find(w) != vertex_to_index.end()) {
                    size_t j = vertex_to_index[w];
                    adj_matrix[i][j] = true;
                    adj_matrix[j][i] = true;  // 无向图，对称设置
                }
            }
        }
        
        // 查找连通组件
        std::vector<bool> visited(index_to_vertex.size(), false);
        std::vector<std::vector<size_t>> components;
        
        for (size_t i = 0; i < index_to_vertex.size(); ++i) {
            if (!visited[i]) {
                std::vector<size_t> component;
                std::queue<size_t> q;
                
                q.push(i);
                visited[i] = true;
                
                while (!q.empty()) {
                    size_t curr = q.front();
                    q.pop();
                    component.push_back(curr);
                    
                    for (size_t j = 0; j < index_to_vertex.size(); ++j) {
                        if (adj_matrix[curr][j] && !visited[j]) {
                            q.push(j);
                            visited[j] = true;
                        }
                    }
                }
                
                components.push_back(component);
            }
        }
        
        // 验证每个连通组件是否形成2-core
        std::vector<bool> forms_2_core(components.size(), false);
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < components.size(); ++i) {
            const auto& component = components[i];
            bool is_2_core = true;
            
            // 检查组件中每个顶点在组件内的度数是否至少为2
            for (size_t idx : component) {
                int internal_degree = 0;
                
                for (size_t other_idx : component) {
                    if (idx != other_idx && adj_matrix[idx][other_idx]) {
                        internal_degree++;
                    }
                }
                
                // 如果有高核心度邻居，也计入
                vertex_t v = index_to_vertex[idx];
                for (vertex_t w : G.adj.neighbors(v)) {
                    if (G.core[w] > 1) {
                        internal_degree++;
                    }
                }
                
                if (internal_degree < 2) {
                    is_2_core = false;
                    break;
                }
            }
            
            forms_2_core[i] = is_2_core;
        }
        
        // 收集所有可以提升的顶点
        for (size_t i = 0; i < components.size(); ++i) {
            if (forms_2_core[i]) {
                for (size_t idx : components[i]) {
                    vertex_t v = index_to_vertex[idx];
                    G.core[v] = 2;
                    promoted.insert(v);
                    
                    // 更新邻居的r值
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] < 2) {
                            ucr_values[w].r++;
                        }
                    }
                }
            }
        }
        
        // 单独处理与高核心度顶点相连的顶点
        for (size_t i = 0; i < core_1_candidates.size(); ++i) {
            // 跳过已经处理过的顶点
            if (potential_promotion[i] && vertex_to_index.find(core_1_candidates[i]) != vertex_to_index.end()) {
                continue;
            }
            
            vertex_t v = core_1_candidates[i];
            int higher_core_neighbors = 0;
            
            for (vertex_t w : G.adj.neighbors(v)) {
                if (G.core[w] >= 2) {
                    higher_core_neighbors++;
                }
            }
            
            if (higher_core_neighbors >= 2) {
                G.core[v] = 2;
                promoted.insert(v);
                
                // 更新邻居的r值
                for (vertex_t w : G.adj.neighbors(v)) {
                    if (G.core[w] < 2) {
                        ucr_values[w].r++;
                    }
                }
            }
        }
    }
    
    double batch_insert(const std::vector<edge_t>& edges) {
        Timer timer;
        
        if (edges.empty()) return 0.0;
        
        // 预处理：找出最大顶点ID
        vertex_t max_vertex_id = 0;
        for (const auto& edge : edges) {
            max_vertex_id = std::max(max_vertex_id, std::max(edge.first, edge.second));
        }
        
        // 确保图和UCR数据结构有足够空间
        if (max_vertex_id >= G.core.size()) {
            G.core.resize(max_vertex_id + 1, 0);
        }
        if (max_vertex_id >= ucr_values.size()) {
            ucr_values.resize(max_vertex_id + 1, CompactUCR(0, 0));
            is_candidate.resize(max_vertex_id + 1, false);
        }
        
        // 更新边和r值
        std::vector<vertex_t> affected_vertices;
        affected_vertices.reserve(edges.size() * 2); // 预分配空间
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (G.adj.neighbors(u).size() == 0 || G.adj.neighbors(v).size() == 0) {
                // 新顶点，避免重新计算整个图
                if (G.adj.neighbors(u).size() == 0) G.core[u] = 1;
                if (G.adj.neighbors(v).size() == 0) G.core[v] = 1;
            }
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv) {
                ucr_values[u].r++;
                affected_vertices.push_back(u);
            } else if (ku > kv) {
                ucr_values[v].r++;
                affected_vertices.push_back(v);
            } else {
                affected_vertices.push_back(u);
                affected_vertices.push_back(v);
            }
        }
        
        // 去重
        std::sort(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.erase(
            std::unique(affected_vertices.begin(), affected_vertices.end()),
            affected_vertices.end()
        );
        
        // 使用边界传播更新s值
        std::unordered_set<vertex_t> update_frontier(affected_vertices.begin(), affected_vertices.end());
        std::unordered_set<vertex_t> next_frontier;
        std::unordered_set<vertex_t> processed;
        
        int max_iterations = 5; // 限制迭代次数
        for (int iter = 0; iter < max_iterations && !update_frontier.empty(); ++iter) {
            for (vertex_t v : update_frontier) {
                if (processed.count(v)) continue;
                processed.insert(v);
                
                int k = G.core[v];
                int old_s = ucr_values[v].s;
                update_s_value(v);
                
                if (old_s != ucr_values[v].s) {
                    // s值变化了，将其同核心度的邻居加入下一轮更新
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] == k && !processed.count(w)) {
                            next_frontier.insert(w);
                        }
                    }
                }
            }
            
            update_frontier = std::move(next_frontier);
            next_frontier.clear();
        }
        
        // 标识并处理候选顶点
        std::unordered_set<vertex_t> candidates;
        for (vertex_t v : processed) {
            if (can_upgrade(v)) {
                candidates.insert(v);
            }
        }
        
        // 特殊处理核心度为1的顶点
        std::unordered_set<vertex_t> promoted_to_2;
        fast_promote_core_1_vertices(candidates, promoted_to_2);
        
        // 从候选集中移除已处理的顶点
        for (vertex_t v : promoted_to_2) {
            candidates.erase(v);
        }
        
        // 处理剩余的候选顶点（核心度>=2）
        if (!candidates.empty()) {
            std::vector<std::vector<vertex_t>> components;
            identify_connected_components(candidates, components);
            
            std::vector<std::unordered_set<vertex_t>> upgraded_vertices_list(components.size());
            
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < components.size(); i++) {
                process_component_for_insertion(components[i], upgraded_vertices_list[i]);
            }
            
            // 合并结果并更新核心度
            std::unordered_set<vertex_t> all_upgraded;
            for (const auto& upgraded : upgraded_vertices_list) {
                all_upgraded.insert(upgraded.begin(), upgraded.end());
            }
            
            // 更新核心度
            for (vertex_t v : all_upgraded) {
                G.core[v]++;
            }
            
            // 如果有顶点升级，更新UCR值
            if (!all_upgraded.empty()) {
                for (vertex_t v : all_upgraded) {
                    // 更新被提升顶点的邻居的r值
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] < G.core[v]) {
                            ucr_values[w].r++;
                        }
                    }
                }
                
                // 增量更新s值，仅处理可能受影响的顶点
                std::unordered_set<vertex_t> update_set;
                for (vertex_t v : all_upgraded) {
                    for (vertex_t w : G.adj.neighbors(v)) {
                        update_set.insert(w);
                        for (vertex_t z : G.adj.neighbors(w)) {
                            if (G.core[z] == G.core[w]) {
                                update_set.insert(z);
                            }
                        }
                    }
                }
                
                // 更新s值
                std::vector<vertex_t> update_vec(update_set.begin(), update_set.end());
                #pragma omp parallel for schedule(dynamic, 1000)
                for (size_t i = 0; i < update_vec.size(); ++i) {
                    update_s_value(update_vec[i]);
                }
            }
        }
        
        return timer.elapsed_milliseconds();
    }
    
    void identify_connected_components(const std::unordered_set<vertex_t>& candidates,
                                      std::vector<std::vector<vertex_t>>& components) {
        std::unordered_set<vertex_t> visited;
        
        // 为每个核心度创建临时映射
        std::unordered_map<int, std::vector<vertex_t>> core_candidates;
        for (vertex_t v : candidates) {
            int k = G.core[v];
            if (k >= 2) { // 只处理核心度>=2的顶点
                core_candidates[k].push_back(v);
            }
        }
        
        // 对每个核心度单独处理
        for (auto& pair : core_candidates) {
            int k = pair.first;
            const std::vector<vertex_t>& k_candidates = pair.second;
            
            for (vertex_t v : k_candidates) {
                if (visited.count(v)) continue;
                
                std::vector<vertex_t> component;
                std::queue<vertex_t> q;
                q.push(v);
                visited.insert(v);
                
                while (!q.empty()) {
                    vertex_t u = q.front();
                    q.pop();
                    
                    if (G.core[u] == k) {
                        component.push_back(u);
                        
                        for (vertex_t w : G.adj.neighbors(u)) {
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
    }
    
    void process_component_for_insertion(const std::vector<vertex_t>& component,
                                        std::unordered_set<vertex_t>& upgraded) {
        if (component.empty()) return;
        
        int k = G.core[component[0]];
        
        std::unordered_set<vertex_t> connected_core;
        for (vertex_t v : component) {
            if (G.core[v] == k && can_upgrade(v)) {
                connected_core.insert(v);
            }
        }
        
        std::unordered_map<vertex_t, int> cd;
        for (vertex_t v : connected_core) {
            cd[v] = ucr_values[v].r + ucr_values[v].s;
        }
        
        std::unordered_set<vertex_t> evicted;
        bool stable = false;
        
        // 模拟级联效应
        while (!stable) {
            stable = true;
            std::vector<vertex_t> to_evict;
            
            for (vertex_t v : connected_core) {
                if (evicted.count(v)) continue;
                
                if (cd[v] <= k) {
                    to_evict.push_back(v);
                    stable = false;
                }
            }
            
            for (vertex_t v : to_evict) {
                evicted.insert(v);
                
                for (vertex_t w : G.adj.neighbors(v)) {
                    if (connected_core.count(w) && !evicted.count(w) && 
                        G.core[w] == k && is_qualified(v)) {
                        cd[w]--;
                    }
                }
            }
        }
        
        for (vertex_t v : connected_core) {
            if (!evicted.count(v)) {
                upgraded.insert(v);
            }
        }
    }
    
    double batch_remove(const std::vector<edge_t>& edges) {
        Timer timer;
        
        if (edges.empty()) return 0.0;
        
        // 预处理：收集受影响的顶点和边
        std::unordered_map<int, std::unordered_set<vertex_t>> k_groups;
        std::vector<vertex_t> affected_vertices;
        affected_vertices.reserve(edges.size() * 2);
        
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u >= G.n() || v >= G.n()) continue;
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            // 预先更新r值
            if (ku < kv) {
                ucr_values[u].r--;
            } else if (ku > kv) {
                ucr_values[v].r--;
            }
            
            affected_vertices.push_back(u);
            affected_vertices.push_back(v);
            
            if (ku > 0) k_groups[ku].insert(u);
            if (kv > 0) k_groups[kv].insert(v);
        }
        
        // 删除边
        for (const auto& edge : edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.n() && v < G.n()) {
                G.remove_edge(u, v);
            }
        }
        
        // 去重并预处理受影响顶点
        std::sort(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.erase(
            std::unique(affected_vertices.begin(), affected_vertices.end()),
            affected_vertices.end()
        );
        
        // 使用边界传播更新s值
        std::unordered_set<vertex_t> update_frontier(affected_vertices.begin(), affected_vertices.end());
        std::unordered_set<vertex_t> next_frontier;
        std::unordered_set<vertex_t> processed;
        
        int max_iterations = 5; // 限制迭代次数
        for (int iter = 0; iter < max_iterations && !update_frontier.empty(); ++iter) {
            for (vertex_t v : update_frontier) {
                if (processed.count(v)) continue;
                processed.insert(v);
                
                int k = G.core[v];
                int old_s = ucr_values[v].s;
                update_s_value(v);
                
                if (old_s != ucr_values[v].s) {
                    // s值变化了，将其同核心度的邻居加入下一轮更新
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] == k && !processed.count(w)) {
                            next_frontier.insert(w);
                        }
                    }
                }
            }
            
            update_frontier = std::move(next_frontier);
            next_frontier.clear();
        }
        
        // 按核心度升序处理
        std::vector<int> k_values;
        for (const auto& pair : k_groups) {
            k_values.push_back(pair.first);
        }
        std::sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            std::vector<vertex_t> seeds;
            for (vertex_t v : k_groups[k]) {
                if (G.core[v] == k) { // 可能在之前的迭代中已经被降级
                    seeds.push_back(v);
                }
            }
            
            if (!seeds.empty()) {
                process_core_level_for_deletion(k, seeds);
            }
        }
        
        return timer.elapsed_milliseconds();
    }
    
    void process_core_level_for_deletion(int k, const std::vector<vertex_t>& seeds) {
       // 找出所有k核连通分量
       std::vector<std::vector<vertex_t>> components;
       std::unordered_set<vertex_t> visited;
       
       for (vertex_t seed : seeds) {
           if (visited.count(seed) || G.core[seed] != k) continue;
           
           std::vector<vertex_t> component;
           std::queue<vertex_t> q;
           q.push(seed);
           visited.insert(seed);
           
           while (!q.empty()) {
               vertex_t v = q.front();
               q.pop();
               
               if (G.core[v] == k) {
                   component.push_back(v);
                   
                   for (vertex_t w : G.adj.neighbors(v)) {
                       if (G.core[w] == k && !visited.count(w)) {
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
       
       // 并行处理每个连通分量
       if (components.size() > 1) {
           #pragma omp parallel for schedule(dynamic, 1)
           for (size_t i = 0; i < components.size(); i++) {
               process_component_for_deletion(k, components[i]);
           }
       } else if (components.size() == 1) {
           // 对于单个大连通分量，避免并行开销
           process_component_for_deletion(k, components[0]);
       }
   }
   
   void process_component_for_deletion(int k, const std::vector<vertex_t>& component) {
       // 计算有效度数
       std::unordered_map<vertex_t, int> eff_deg;
       std::vector<vertex_t> initial_degrade;
       
       for (vertex_t v : component) {
           int count = 0;
           for (vertex_t w : G.adj.neighbors(v)) {
               if (G.core[w] >= k) count++;
           }
           eff_deg[v] = count;
           
           if (count < k) {
               initial_degrade.push_back(v);
           }
       }
       
       // 如果没有需要降级的顶点，直接返回
       if (initial_degrade.empty()) return;
       
       // 使用双端队列优化级联降级
       std::deque<vertex_t> degrade_queue(initial_degrade.begin(), initial_degrade.end());
       std::unordered_set<vertex_t> degraded;
       
       while (!degrade_queue.empty()) {
           vertex_t v = degrade_queue.front();
           degrade_queue.pop_front();
           
           if (G.core[v] != k || degraded.count(v)) continue;
           
           // 降级顶点
           G.core[v] = k - 1;
           degraded.insert(v);
           
           // 更新邻居的有效度数
           for (vertex_t w : G.adj.neighbors(v)) {
               if (G.core[w] == k && !degraded.count(w)) {
                   eff_deg[w]--;
                   if (eff_deg[w] < k) {
                       degrade_queue.push_back(w);
                   }
               }
           }
       }
       
       // 更新被降级顶点的邻居的r值
       for (vertex_t v : degraded) {
           for (vertex_t w : G.adj.neighbors(v)) {
               if (G.core[w] > k - 1) {
                   // 顶点w的邻居v从k降到k-1
                   ucr_values[w].r--;
               }
           }
       }
       
       // 增量更新s值
       std::unordered_set<vertex_t> update_set;
       for (vertex_t v : degraded) {
           for (vertex_t w : G.adj.neighbors(v)) {
               update_set.insert(w);
           }
       }
       
       // 仅处理可能受影响的顶点
       std::vector<vertex_t> update_vec(update_set.begin(), update_set.end());
       #pragma omp parallel for schedule(dynamic, 1000)
       for (size_t i = 0; i < update_vec.size(); ++i) {
           update_s_value(update_vec[i]);
       }
   }
   
   size_t estimate_memory_usage() const {
       size_t total = 0;
       
       total += G.memory_usage();
       total += ucr_values.capacity() * sizeof(CompactUCR);
       total += is_candidate.capacity() * sizeof(bool);
       
       for (const auto& vec : core_level_vertices) {
           total += vec.capacity() * sizeof(vertex_t);
       }
       
       return total;
   }
};

// MCD/PCD方法的优化实现
class OptimizedMCDPCDCore {
protected:
   Graph& G;
   std::vector<int> MCD;
   std::vector<int> PCD;
   
public:
   OptimizedMCDPCDCore(Graph& g) : G(g) {
       reset(true);
   }
   
   void reset(bool full_reset = false) {
       size_t n = G.n();
       
       if (MCD.size() < n) {
           MCD.resize(n, 0);
       }
       
       if (PCD.size() < n) {
           PCD.resize(n, 0);
       }
       
       if (full_reset) {
           // 重新计算MCD
           #pragma omp parallel for schedule(dynamic, 1000)
           for (vertex_t v = 0; v < n; ++v) {
               int count = 0;
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (G.core[w] >= G.core[v]) {
                       ++count;
                   }
               }
               MCD[v] = count;
           }
           
           // 重新计算PCD
           #pragma omp parallel for schedule(dynamic, 1000)
           for (vertex_t v = 0; v < n; ++v) {
               int count = 0;
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (qual(v, w)) {
                       ++count;
                   }
               }
               PCD[v] = count;
           }
       }
   }
   
   bool qual(vertex_t v, vertex_t w) const {
       if (v >= G.n() || w >= G.n()) return false;
       
       int k = G.core[v];
       return G.core[w] > k || (G.core[w] == k && MCD[w] > k);
   }
   
   double batch_insert(const std::vector<edge_t>& edges) {
       Timer timer;
       
       if (edges.empty()) return 0.0;
       
       // 预处理：找出最大顶点ID
       vertex_t max_vertex_id = 0;
       for (const auto& edge : edges) {
           max_vertex_id = std::max(max_vertex_id, std::max(edge.first, edge.second));
       }
       
       // 确保图和MCD/PCD数据结构有足够空间
       if (max_vertex_id >= G.core.size()) {
           G.core.resize(max_vertex_id + 1, 0);
       }
       if (max_vertex_id >= MCD.size()) {
           MCD.resize(max_vertex_id + 1, 0);
           PCD.resize(max_vertex_id + 1, 0);
       }
       
       // 收集受影响的顶点
       std::unordered_set<vertex_t> affected_vertices;
       
       for (const auto& edge : edges) {
           vertex_t u = edge.first;
           vertex_t v = edge.second;
           
           if (G.adj.neighbors(u).size() == 0 || G.adj.neighbors(v).size() == 0) {
               // 新顶点，避免重新计算整个图
               if (G.adj.neighbors(u).size() == 0) G.core[u] = 1;
               if (G.adj.neighbors(v).size() == 0) G.core[v] = 1;
           }
           
           G.add_edge(u, v);
           
           // 更新MCD
           if (G.core[v] >= G.core[u]) {
               ++MCD[u];
           }
           if (G.core[u] >= G.core[v]) {
               ++MCD[v];
           }
           
           affected_vertices.insert(u);
           affected_vertices.insert(v);
           
           // 添加邻居也可能受影响
           for (vertex_t w : G.adj.neighbors(u)) {
               affected_vertices.insert(w);
           }
           for (vertex_t w : G.adj.neighbors(v)) {
               affected_vertices.insert(w);
           }
       }
       
       // 更新PCD - 只更新受影响的顶点
       update_PCD_for_vertices(affected_vertices);
       
       // 处理可能需要升级的顶点
       for (size_t i = 0; i < edges.size(); ++i) {
           vertex_t u = edges[i].first;
           vertex_t v = edges[i].second;
           
           std::unordered_set<vertex_t> candidates;
           
           if (PCD[u] > G.core[u]) {
               candidates.insert(u);
           }
           if (PCD[v] > G.core[v]) {
               candidates.insert(v);
           }
           
           process_core_update(candidates);
           
           // 如果有顶点升级，更新MCD和PCD
           if (!candidates.empty()) {
               update_PCD_for_vertices(affected_vertices);
           }
       }
       
       return timer.elapsed_milliseconds();
   }
   
   double batch_remove(const std::vector<edge_t>& edges) {
       Timer timer;
       
       if (edges.empty()) return 0.0;
       
       // 收集受影响的顶点
       std::unordered_set<vertex_t> affected_vertices;
       std::unordered_map<int, std::vector<vertex_t>> k_groups;
       
       for (const auto& edge : edges) {
           vertex_t u = edge.first;
           vertex_t v = edge.second;
           
           if (u >= G.n() || v >= G.n()) continue;
           
           affected_vertices.insert(u);
           affected_vertices.insert(v);
           
           // 添加邻居也可能受影响
           for (vertex_t w : G.adj.neighbors(u)) {
               affected_vertices.insert(w);
           }
           for (vertex_t w : G.adj.neighbors(v)) {
               affected_vertices.insert(w);
           }
           
           // 更新MCD
           if (G.core[v] >= G.core[u]) {
               --MCD[u];
           }
           if (G.core[u] >= G.core[v]) {
               --MCD[v];
           }
           
           // 按核心度分组
           int ku = G.core[u];
           int kv = G.core[v];
           
           if (ku > 0) k_groups[ku].push_back(u);
           if (kv > 0) k_groups[kv].push_back(v);
           
           // 删除边
           G.remove_edge(u, v);
       }
       
       // 更新PCD
       update_PCD_for_vertices(affected_vertices);
       
       // 按核心度升序处理
       std::vector<int> k_values;
       for (const auto& pair : k_groups) {
           k_values.push_back(pair.first);
       }
       std::sort(k_values.begin(), k_values.end());
       
       for (int k : k_values) {
           process_core_level_removal(k, k_groups[k]);
           
           // 更新MCD和PCD
           update_PCD_for_vertices(affected_vertices);
       }
       
       return timer.elapsed_milliseconds();
   }
   
   void update_PCD_for_vertices(const std::unordered_set<vertex_t>& vertices) {
       std::vector<vertex_t> vec(vertices.begin(), vertices.end());
       
       #pragma omp parallel for schedule(dynamic, 1000)
       for (size_t i = 0; i < vec.size(); ++i) {
           vertex_t v = vec[i];
           int count = 0;
           
           for (vertex_t w : G.adj.neighbors(v)) {
               if (qual(v, w)) {
                   ++count;
               }
           }
           
           PCD[v] = count;
       }
   }
   
   void process_core_update(const std::unordered_set<vertex_t>& candidates) {
       if (candidates.empty()) return;
       
       std::unordered_set<vertex_t> upgrade_vertices;
       
       for (vertex_t v : candidates) {
           int k = G.core[v];
           
           if (PCD[v] <= k) continue;
           
           std::unordered_set<vertex_t> connected_core;
           std::unordered_set<vertex_t> evicted;
           
           // 找出连通的k-核
           {
               std::queue<vertex_t> q;
               std::unordered_set<vertex_t> seen;
               q.push(v);
               seen.insert(v);
               
               while (!q.empty()) {
                   vertex_t x = q.front();
                   q.pop();
                   
                   if (G.core[x] == k) {
                       connected_core.insert(x);
                       
                       for (vertex_t w : G.adj.neighbors(x)) {
                           if (G.core[w] == k && !seen.count(w)) {
                               seen.insert(w);
                               q.push(w);
                           }
                       }
                   }
               }
           }
           
           // 计算cd值
           std::unordered_map<vertex_t, int> cd;
           for (vertex_t x : connected_core) {
               cd[x] = PCD[x];
           }
           
           // 模拟级联效应
           bool stable = false;
           while (!stable) {
               stable = true;
               std::vector<vertex_t> to_evict;
               
               for (vertex_t x : connected_core) {
                   if (evicted.count(x)) continue;
                   
                   if (cd[x] <= k) {
                       to_evict.push_back(x);
                       stable = false;
                   }
               }
               
               for (vertex_t x : to_evict) {
                   evicted.insert(x);
                   
                   for (vertex_t w : G.adj.neighbors(x)) {
                       if (connected_core.count(w) && !evicted.count(w) && 
                           G.core[w] == k && qual(w, x)) {
                           cd[w]--;
                       }
                   }
               }
           }
           
           // 添加不被淘汰的顶点到升级集合
           for (vertex_t x : connected_core) {
               if (!evicted.count(x)) {
                   upgrade_vertices.insert(x);
               }
           }
       }
       
       // 更新核心度
       for (vertex_t v : upgrade_vertices) {
           ++G.core[v];
       }
       
       // 如果有顶点升级，更新MCD
       if (!upgrade_vertices.empty()) {
           std::unordered_set<vertex_t> update_set;
           
           for (vertex_t v : upgrade_vertices) {
               update_set.insert(v);
               
               for (vertex_t w : G.adj.neighbors(v)) {
                   update_set.insert(w);
               }
           }
           
           std::vector<vertex_t> update_vec(update_set.begin(), update_set.end());
           
           #pragma omp parallel for schedule(dynamic, 1000)
           for (size_t i = 0; i < update_vec.size(); ++i) {
               vertex_t v = update_vec[i];
               int count = 0;
               
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (G.core[w] >= G.core[v]) {
                       ++count;
                   }
               }
               
               MCD[v] = count;
           }
       }
   }
   
   void process_core_level_removal(int k, const std::vector<vertex_t>& seeds) {
       std::vector<vertex_t> candidates;
       std::unordered_map<vertex_t, int> eff_deg;
       std::unordered_set<vertex_t> visited;
       
       // 找出所有k-核连通分量
       for (vertex_t seed : seeds) {
           if (visited.count(seed) || G.core[seed] != k) continue;
           
           std::queue<vertex_t> q;
           q.push(seed);
           visited.insert(seed);
           
           while (!q.empty()) {
               vertex_t v = q.front();
               q.pop();
               
               if (G.core[v] == k) {
                   candidates.push_back(v);
                   
                   int count = 0;
                   for (vertex_t w : G.adj.neighbors(v)) {
                       if (G.core[w] >= k) count++;
                   }
                   eff_deg[v] = count;
                   
                   for (vertex_t w : G.adj.neighbors(v)) {
                       if (G.core[w] == k && !visited.count(w)) {
                           visited.insert(w);
                           q.push(w);
                       }
                   }
               }
           }
       }
       
       // 如果没有候选顶点，直接返回
       if (candidates.empty()) return;
       
       // 找出初始降级顶点
       std::deque<vertex_t> degrade_queue;
       std::unordered_set<vertex_t> degraded;
       
       for (vertex_t v : candidates) {
           if (eff_deg[v] < k) {
               degrade_queue.push_back(v);
           }
       }
       
       // 级联降级
       while (!degrade_queue.empty()) {
           vertex_t v = degrade_queue.front();
           degrade_queue.pop_front();
           
           if (G.core[v] != k || degraded.count(v)) continue;
           
           G.core[v] = k - 1;
           degraded.insert(v);
           
           for (vertex_t w : G.adj.neighbors(v)) {
               if (G.core[w] == k) {
                   eff_deg[w]--;
                   if (eff_deg[w] < k) {
                       degrade_queue.push_back(w);
                   }
               }
           }
       }
       
       // 更新MCD
       if (!degraded.empty()) {
           std::unordered_set<vertex_t> update_set;
           
           for (vertex_t v : degraded) {
               update_set.insert(v);
               
               for (vertex_t w : G.adj.neighbors(v)) {
                   update_set.insert(w);
               }
           }
           
           std::vector<vertex_t> update_vec(update_set.begin(), update_set.end());
           
           #pragma omp parallel for schedule(dynamic, 1000)
           for (size_t i = 0; i < update_vec.size(); ++i) {
               vertex_t v = update_vec[i];
               int count = 0;
               
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (G.core[w] >= G.core[v]) {
                       ++count;
                   }
               }
               
               MCD[v] = count;
           }
       }
   }
   
   size_t estimate_memory_usage() const {
       size_t total = 0;
       
       total += G.memory_usage();
       total += MCD.capacity() * sizeof(int);
       total += PCD.capacity() * sizeof(int);
       
       return total;
   }
};

// 时序图处理器
class TemporalGraphProcessor {
private:
   std::string filepath;
   std::vector<TemporalEdge> edges;
   std::vector<std::vector<edge_t>> bins;
   
   vertex_t max_vertex_id = 0;
   timestamp_t min_timestamp = 0;
   timestamp_t max_timestamp = 0;
   
   void bin_edges(int num_bins) {
       if (edges.empty()) return;
       
       // 获取最小和最大时间戳
       min_timestamp = edges[0].timestamp;
       max_timestamp = edges[0].timestamp;
       
       for (const auto& edge : edges) {
           min_timestamp = std::min(min_timestamp, edge.timestamp);
           max_timestamp = std::max(max_timestamp, edge.timestamp);
       }
       
       // 计算bin宽度
       double bin_span = static_cast<double>(max_timestamp - min_timestamp + 1) / num_bins;
       
       bins.resize(num_bins);
       
       // 分配边到bin
       #pragma omp parallel
       {
           // 每个线程的局部bins
           std::vector<std::vector<edge_t>> local_bins(num_bins);
           
           #pragma omp for
           for (size_t i = 0; i < edges.size(); ++i) {
               const auto& edge = edges[i];
               int bin_idx = std::min(static_cast<int>((edge.timestamp - min_timestamp) / bin_span), num_bins - 1);
               local_bins[bin_idx].push_back({edge.src, edge.dst});
           }
           
           // 合并局部bins到全局bins
           #pragma omp critical
           {
               for (int i = 0; i < num_bins; ++i) {
                   bins[i].insert(bins[i].end(), local_bins[i].begin(), local_bins[i].end());
               }
           }
       }
       
       std::cout << "Edges binned into " << num_bins << " bins. Bin span: " << bin_span << "ms" << std::endl;
       
       // 打印每个bin的边数
       for (int i = 0; i < std::min(10, num_bins); ++i) {
           std::cout << "Bin " << i << ": " << bins[i].size() << " edges" << std::endl;
       }
       std::cout << "..." << std::endl;
   }
   
public:
   TemporalGraphProcessor(const std::string& path) : filepath(path) {
       load_data();
   }
   
   void load_data() {
       Timer timer;
       std::ifstream file(filepath);
       if (!file.is_open()) {
           std::cerr << "Failed to open file: " << filepath << std::endl;
           return;
       }
       
       // 预分配内存以提高性能
       const size_t CHUNK_SIZE = 100000;
       edges.reserve(CHUNK_SIZE);
       
       std::string line;
       while (std::getline(file, line)) {
           // 跳过注释行
           if (line.empty() || line[0] == '%' || line[0] == '#') continue;
           
           std::istringstream iss(line);
           vertex_t src, dst;
           timestamp_t ts;
           
           if (iss >> src >> dst >> ts) {
               edges.emplace_back(src, dst, ts);
               max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
               
               // 如果vector达到容量，再次预分配
               if (edges.size() == edges.capacity()) {
                   edges.reserve(edges.size() + CHUNK_SIZE);
               }
           }
       }
       
       std::cout << "Loaded " << edges.size() << " edges in " << timer.elapsed_seconds() 
                 << " seconds. Max vertex ID: " << max_vertex_id << std::endl;
   }
   
   void create_bins(int num_bins) {
       Timer timer;
       bin_edges(num_bins);
       std::cout << "Binning completed in " << timer.elapsed_seconds() << " seconds." << std::endl;
   }
   
   const std::vector<edge_t>& get_bin_edges(int bin_idx) const {
       if (bin_idx < 0 || bin_idx >= bins.size()) {
           static const std::vector<edge_t> empty;
           return empty;
       }
       return bins[bin_idx];
   }
   
   size_t get_num_bins() const {
       return bins.size();
   }
   
   vertex_t get_max_vertex_id() const {
       return max_vertex_id;
   }

   timestamp_t get_min_timestamp() const {
       return min_timestamp;
   }
   
   timestamp_t get_max_timestamp() const {
       return max_timestamp;
   }
};

// 实验运行器
class ExperimentRunner {
private:
   TemporalGraphProcessor& processor;
   int window_size;
   int starting_bin;
   int num_steps;
   int max_threads;
   
public:
   ExperimentRunner(TemporalGraphProcessor& proc, int win_size, int start_bin, int steps, int threads)
       : processor(proc), window_size(win_size), starting_bin(start_bin), num_steps(steps), max_threads(threads) {}
   
   void run() {
       // 设置OpenMP线程数
       omp_set_num_threads(max_threads);
       std::cout << "Using " << max_threads << " OpenMP threads" << std::endl;
       
       // 初始化图
       Graph graph;
       
       // 初始化两种核心度计算方法
       OptimizedMCDPCDCore mcd_pcd_core(graph);
       OptimizedUCRCore ucr_core(graph);
       
       // 获取所有bins
       size_t num_bins = processor.get_num_bins();
       if (starting_bin + window_size > num_bins) {
           std::cerr << "Invalid starting bin or window size!" << std::endl;
           return;
       }
       
       // 构建初始窗口
       Timer window_timer;
       std::cout << "Building initial window (bins " << starting_bin << " to " 
                 << (starting_bin + window_size - 1) << ")..." << std::endl;
       
       std::vector<edge_t> all_edges;
       size_t total_edges = 0;
       
       for (int i = starting_bin; i < starting_bin + window_size; ++i) {
           const auto& bin_edges = processor.get_bin_edges(i);
           total_edges += bin_edges.size();
       }
       
       all_edges.reserve(total_edges);
       
       for (int i = starting_bin; i < starting_bin + window_size; ++i) {
           const auto& bin_edges = processor.get_bin_edges(i);
           all_edges.insert(all_edges.end(), bin_edges.begin(), bin_edges.end());
       }
       
       // 添加初始边集
       Timer add_edges_timer;
       for (const auto& edge : all_edges) {
           graph.add_edge(edge.first, edge.second);
       }
       std::cout << "Added " << all_edges.size() << " edges in " << add_edges_timer.elapsed_seconds() << " seconds" << std::endl;
       
       // 计算初始核心度
       Timer core_timer;
       graph.compute_core_numbers();
       std::cout << "Computed initial core numbers in " << core_timer.elapsed_seconds() << " seconds" << std::endl;
       
       // 初始化两种方法的状态
       Timer reset_timer;
       mcd_pcd_core.reset(true);
       ucr_core.reset(true);
       std::cout << "Reset core methods in " << reset_timer.elapsed_seconds() << " seconds" << std::endl;
       
       // 估计初始内存使用
       size_t mcd_pcd_memory = mcd_pcd_core.estimate_memory_usage();
       size_t ucr_memory = ucr_core.estimate_memory_usage();
       
       std::cout << "Initial window built in " << window_timer.elapsed_seconds() << " seconds. Total edges: " << all_edges.size() << std::endl;
       std::cout << "MCD/PCD memory usage: " << (mcd_pcd_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
       std::cout << "UCR memory usage: " << (ucr_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
       std::cout << "Memory reduction: " << (100.0 * (1.0 - (double)ucr_memory / mcd_pcd_memory)) << "%" << std::endl;
       
       // 结果汇总
       struct StepResult {
           int step;
           int remove_bin;
           int add_bin;
           size_t num_remove_edges;
           size_t num_add_edges;
           double mcd_pcd_remove_time;
           double mcd_pcd_add_time;
           double ucr_remove_time;
           double ucr_add_time;
       };
       
       std::vector<StepResult> results;
       results.reserve(num_steps);
       
       // 开始滑动窗口实验
       std::cout << "\nStarting sliding window experiment..." << std::endl;
       
       for (int step = 0; step < num_steps; ++step) {
           int remove_bin = starting_bin + step;
           int add_bin = starting_bin + window_size + step;
           
           if (add_bin >= num_bins) break;
           
           std::cout << "\nStep " << step + 1 << "/" << num_steps 
                     << ": Remove bin " << remove_bin << ", Add bin " << add_bin << std::endl;
           
           // 获取要删除和添加的边
           const auto& remove_edges = processor.get_bin_edges(remove_bin);
           const auto& add_edges = processor.get_bin_edges(add_bin);
           
           std::cout << "Removing " << remove_edges.size() << " edges, Adding " << add_edges.size() << " edges" << std::endl;
           
           // 测试MCD/PCD方法
           Graph graph_copy = graph;
           OptimizedMCDPCDCore mcd_pcd_copy(graph_copy);
           mcd_pcd_copy.reset(true);
           
           std::cout << "Testing MCD/PCD method..." << std::endl;
           double mcd_pcd_remove_time = mcd_pcd_copy.batch_remove(remove_edges);
           double mcd_pcd_add_time = mcd_pcd_copy.batch_insert(add_edges);
           
           // 测试UCR方法
           Graph graph_copy2 = graph;
           OptimizedUCRCore ucr_copy(graph_copy2);
           ucr_copy.reset(true);
           
           std::cout << "Testing UCR method..." << std::endl;
           double ucr_remove_time = ucr_copy.batch_remove(remove_edges);
           double ucr_add_time = ucr_copy.batch_insert(add_edges);
           
           // 更新原图
           Timer update_timer;
           for (const auto& edge : remove_edges) {
               graph.remove_edge(edge.first, edge.second);
           }
           for (const auto& edge : add_edges) {
               graph.add_edge(edge.first, edge.second);
           }
           
           // 更新原始核心度计算
           graph.compute_core_numbers();
           mcd_pcd_core.reset(true);
           ucr_core.reset(true);
           std::cout << "Updated original graph in " << update_timer.elapsed_seconds() << " seconds" << std::endl;
           
           // 存储结果
           results.push_back({
               step + 1,
               remove_bin,
               add_bin,
               remove_edges.size(),
               add_edges.size(),
               mcd_pcd_remove_time,
               mcd_pcd_add_time,
               ucr_remove_time,
               ucr_add_time
           });
           
           // 打印当前步骤的结果
           std::cout << "MCD/PCD Remove time: " << mcd_pcd_remove_time << " ms" << std::endl;
           std::cout << "MCD/PCD Add time: " << mcd_pcd_add_time << " ms" << std::endl;
           std::cout << "UCR Remove time: " << ucr_remove_time << " ms" << std::endl;
           std::cout << "UCR Add time: " << ucr_add_time << " ms" << std::endl;
           
           // 计算加速比
           double remove_speedup = mcd_pcd_remove_time / ucr_remove_time;
           double add_speedup = mcd_pcd_add_time / ucr_add_time;
           double total_speedup = (mcd_pcd_remove_time + mcd_pcd_add_time) / (ucr_remove_time + ucr_add_time);
           
           std::cout << "Remove speedup: " << remove_speedup << "x" << std::endl;
           std::cout << "Add speedup: " << add_speedup << "x" << std::endl;
           std::cout << "Total speedup: " << total_speedup << "x" << std::endl;
       }
       
       // 打印总结
       std::cout << "\n=== Experiment Summary ===" << std::endl;
       
       // 计算平均值
       double avg_mcd_pcd_remove_time = 0.0;
       double avg_mcd_pcd_add_time = 0.0;
       double avg_ucr_remove_time = 0.0;
       double avg_ucr_add_time = 0.0;
       
       for (const auto& result : results) {
           avg_mcd_pcd_remove_time += result.mcd_pcd_remove_time;
           avg_mcd_pcd_add_time += result.mcd_pcd_add_time;
           avg_ucr_remove_time += result.ucr_remove_time;
           avg_ucr_add_time += result.ucr_add_time;
       }
       
       if (!results.empty()) {
           avg_mcd_pcd_remove_time /= results.size();
           avg_mcd_pcd_add_time /= results.size();
           avg_ucr_remove_time /= results.size();
           avg_ucr_add_time /= results.size();
           
           double avg_remove_speedup = avg_mcd_pcd_remove_time / avg_ucr_remove_time;
           double avg_add_speedup = avg_mcd_pcd_add_time / avg_ucr_add_time;
           double avg_total_speedup = (avg_mcd_pcd_remove_time + avg_mcd_pcd_add_time) / 
                                    (avg_ucr_remove_time + avg_ucr_add_time);
           
           std::cout << "Average MCD/PCD Remove time: " << avg_mcd_pcd_remove_time << " ms" << std::endl;
           std::cout << "Average MCD/PCD Add time: " << avg_mcd_pcd_add_time << " ms" << std::endl;
           std::cout << "Average UCR Remove time: " << avg_ucr_remove_time << " ms" << std::endl;
           std::cout << "Average UCR Add time: " << avg_ucr_add_time << " ms" << std::endl;
           std::cout << "Average Remove speedup: " << avg_remove_speedup << "x" << std::endl;
           std::cout << "Average Add speedup: " << avg_add_speedup << "x" << std::endl;
           std::cout << "Average Total speedup: " << avg_total_speedup << "x" << std::endl;
       }
       
       // 打印每个步骤的详细结果
       std::cout << "\nDetailed Results:" << std::endl;
       std::cout << "Step,RemoveBin,AddBin,NumRemoveEdges,NumAddEdges,"
                << "MCD_PCD_RemoveTime,MCD_PCD_AddTime,UCR_RemoveTime,UCR_AddTime,"
                << "RemoveSpeedup,AddSpeedup,TotalSpeedup" << std::endl;
       
       for (const auto& result : results) {
           double remove_speedup = result.mcd_pcd_remove_time / result.ucr_remove_time;
           double add_speedup = result.mcd_pcd_add_time / result.ucr_add_time;
           double total_speedup = (result.mcd_pcd_remove_time + result.mcd_pcd_add_time) / 
                                (result.ucr_remove_time + result.ucr_add_time);
           
           std::cout << result.step << "," 
                    << result.remove_bin << "," 
                    << result.add_bin << "," 
                    << result.num_remove_edges << "," 
                    << result.num_add_edges << "," 
                    << result.mcd_pcd_remove_time << "," 
                    << result.mcd_pcd_add_time << "," 
                    << result.ucr_remove_time << "," 
                    << result.ucr_add_time << "," 
                    << remove_speedup << "," 
                    << add_speedup << "," 
                    << total_speedup << std::endl;
       }
       
       // 将结果保存到CSV文件
       std::string csv_filename = "ucr_vs_mcdpcd_results.csv";
       std::ofstream csv_file(csv_filename);
       
       if (csv_file.is_open()) {
           csv_file << "Step,RemoveBin,AddBin,NumRemoveEdges,NumAddEdges,"
                   << "MCD_PCD_RemoveTime,MCD_PCD_AddTime,UCR_RemoveTime,UCR_AddTime,"
                   << "RemoveSpeedup,AddSpeedup,TotalSpeedup" << std::endl;
           
           for (const auto& result : results) {
               double remove_speedup = result.mcd_pcd_remove_time / result.ucr_remove_time;
               double add_speedup = result.mcd_pcd_add_time / result.ucr_add_time;
               double total_speedup = (result.mcd_pcd_remove_time + result.mcd_pcd_add_time) / 
                                    (result.ucr_remove_time + result.ucr_add_time);
               
               csv_file << result.step << "," 
                       << result.remove_bin << "," 
                       << result.add_bin << "," 
                       << result.num_remove_edges << "," 
                       << result.num_add_edges << "," 
                       << result.mcd_pcd_remove_time << "," 
                       << result.mcd_pcd_add_time << "," 
                       << result.ucr_remove_time << "," 
                       << result.ucr_add_time << "," 
                       << remove_speedup << "," 
                       << add_speedup << "," 
                       << total_speedup << std::endl;
           }
           
           csv_file.close();
           std::cout << "Results saved to " << csv_filename << std::endl;
       } else {
           std::cerr << "Failed to open file for writing: " << csv_filename << std::endl;
       }
   }
};

int main(int argc, char* argv[]) {
   // 默认参数
   std::string filepath = "sx-stackoverflow.txt";
   int num_bins = 1000;
   int starting_bin = 400;
   int num_steps = 10;  // 默认减少步数以加快测试
   int max_threads = omp_get_max_threads();
   int window_size = 100;  // 默认窗口大小

   // 解析命令行参数
   if (argc > 1) filepath = argv[1];
   if (argc > 2) num_bins = std::stoi(argv[2]);
   if (argc > 3) starting_bin = std::stoi(argv[3]);
   if (argc > 4) num_steps = std::stoi(argv[4]);
   if (argc > 5) max_threads = std::stoi(argv[5]);
   if (argc > 6) window_size = std::stoi(argv[6]);

   std::cout << "Configuration:" << std::endl;
   std::cout << "- Dataset: " << filepath << std::endl;
   std::cout << "- Number of bins: " << num_bins << std::endl;
   std::cout << "- Starting bin: " << starting_bin << std::endl;
   std::cout << "- Number of steps: " << num_steps << std::endl;
   std::cout << "- Max threads: " << max_threads << std::endl;
   std::cout << "- Window size: " << window_size << std::endl;

   // 加载时序图
   Timer load_timer;
   std::cout << "Loading temporal graph data..." << std::endl;
   TemporalGraphProcessor processor(filepath);
   std::cout << "Loaded graph data in " << load_timer.elapsed_seconds() << " seconds." << std::endl;

   // 将数据分成指定数量的bins
   Timer bin_timer;
   std::cout << "Creating " << num_bins << " bins..." << std::endl;
   processor.create_bins(num_bins);
   std::cout << "Created bins in " << bin_timer.elapsed_seconds() << " seconds." << std::endl;

   // 创建实验运行器
   std::cout << "Starting experiment..." << std::endl;
   Timer experiment_timer;
   ExperimentRunner experiment(processor, window_size, starting_bin, num_steps, max_threads);

   // 运行实验
   experiment.run();
   std::cout << "Experiment completed in " << experiment_timer.elapsed_seconds() << " seconds." << std::endl;

   return 0;
}