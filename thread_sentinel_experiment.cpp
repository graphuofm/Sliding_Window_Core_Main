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

// 优化的邻接表结构（加入哨兵模式）
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
    
    // 优化版本使用哨兵模式的邻居遍历
    template<typename Func>
    void for_each_neighbor_with_sentinel(vertex_t v, Func f) const {
        if (v >= adj_lists.size()) return;
        
        const auto& neighbors = adj_lists[v];
        if (neighbors.empty()) return;
        
        // 哨兵模式优化：避免在循环中重复检查边界条件
        size_t i = 0;
        size_t n = neighbors.size();
        
        // 直接处理四个一组的块（展开循环）
        for (; i + 3 < n; i += 4) {
            f(neighbors[i]);
            f(neighbors[i+1]);
            f(neighbors[i+2]);
            f(neighbors[i+3]);
        }
        
        // 处理剩余元素
        for (; i < n; ++i) {
            f(neighbors[i]);
        }
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
    
    // BZ算法（剥洋葱）计算核心度
    void compute_core_numbers() {
        size_t n_vertices = adj.size();
        std::fill(core.begin(), core.end(), 0);
        
        if (n_vertices == 0) return;
        
        // 计算最大度数
        int max_degree = 0;
        std::vector<int> degree(n_vertices);
        
        #pragma omp parallel for reduction(max:max_degree)
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
                
                adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                    if (processed[w]) return;
                    
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
                });
            }
        }
    }
    
    size_t memory_usage() const {
        return core.capacity() * sizeof(int) + adj.memory_usage();
    }
};

// 优化的UCR实现，加入哨兵模式
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
    
    // 哨兵优化标志
    bool use_sentinel;
    
public:
    OptimizedUCRCore(Graph& g, bool enable_sentinel = true) 
        : G(g), max_core_level(0), use_sentinel(enable_sentinel) {
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
                
                if (use_sentinel) {
                    // 使用哨兵模式优化
                    G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                        if (G.core[w] > k) {
                            ++r_count;
                        }
                    });
                } else {
                    // 传统方式
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] > k) {
                            ++r_count;
                        }
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
        
        if (use_sentinel) {
            // 使用哨兵模式优化
            G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                if (G.core[w] == k && is_qualified(w)) {
                    ++s_count;
                }
            });
        } else {
            // 传统方式
            for (vertex_t w : G.adj.neighbors(v)) {
                if (G.core[w] == k && is_qualified(w)) {
                    ++s_count;
                }
            }
        }
        
        ucr_values[v].s = s_count;
    }
    
    int calculate_s_value(vertex_t v) {
        int k = G.core[v];
        int s_count = 0;
        
        if (use_sentinel) {
            // 使用哨兵模式优化
            G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                if (G.core[w] == k && is_qualified(w)) {
                    ++s_count;
                }
            });
        } else {
            // 传统方式
            for (vertex_t w : G.adj.neighbors(v)) {
                if (G.core[w] == k && is_qualified(w)) {
                    ++s_count;
                }
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
            if (use_sentinel) {
                // 使用哨兵模式优化
                G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                    if (G.core[w] >= 1) {
                        valid_neighbors++;
                        if (valid_neighbors >= 2) {
                            potential_promotion[i] = true;
                        }
                    }
                });
            } else {
                // 传统方式
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
            
            if (use_sentinel) {
                // 使用哨兵模式优化
                G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                    if (G.core[w] == 1 && vertex_to_index.find(w) != vertex_to_index.end()) {
                        size_t j = vertex_to_index[w];
                        adj_matrix[i][j] = true;
                        adj_matrix[j][i] = true;  // 无向图，对称设置
                    }
                });
            } else {
                // 传统方式
                for (vertex_t w : G.adj.neighbors(v)) {
                    if (G.core[w] == 1 && vertex_to_index.find(w) != vertex_to_index.end()) {
                        size_t j = vertex_to_index[w];
                        adj_matrix[i][j] = true;
                        adj_matrix[j][i] = true;  // 无向图，对称设置
                    }
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
                if (use_sentinel) {
                    // 使用哨兵模式优化
                    G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                        if (G.core[w] > 1) {
                            internal_degree++;
                        }
                    });
                } else {
                    // 传统方式
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] > 1) {
                            internal_degree++;
                        }
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
                    if (use_sentinel) {
                        // 使用哨兵模式优化
                        G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                            if (G.core[w] < 2) {
                                ucr_values[w].r++;
                            }
                        });
                    } else {
                        // 传统方式
                        for (vertex_t w : G.adj.neighbors(v)) {
                            if (G.core[w] < 2) {
                                ucr_values[w].r++;
                            }
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
            
            if (use_sentinel) {
                // 使用哨兵模式优化
                G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                    if (G.core[w] >= 2) {
                        higher_core_neighbors++;
                    }
                });
            } else {
                // 传统方式
                for (vertex_t w : G.adj.neighbors(v)) {
                    if (G.core[w] >= 2) {
                        higher_core_neighbors++;
                    }
                }
            }
            
            if (higher_core_neighbors >= 2) {
                G.core[v] = 2;
                promoted.insert(v);
                
                // 更新邻居的r值
                if (use_sentinel) {
                    // 使用哨兵模式优化
                    G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                        if (G.core[w] < 2) {
                            ucr_values[w].r++;
                        }
                    });
                } else {
                    // 传统方式
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (G.core[w] < 2) {
                            ucr_values[w].r++;
                        }
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
                    if (use_sentinel) {
                        // 使用哨兵模式优化
                        G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                            if (G.core[w] == k && !processed.count(w)) {
                                next_frontier.insert(w);
                            }
                        });
                    } else {
                        // 传统方式
                        for (vertex_t w : G.adj.neighbors(v)) {
                            if (G.core[w] == k && !processed.count(w)) {
                                next_frontier.insert(w);
                            }
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
                    if (use_sentinel) {
                        // 使用哨兵模式优化
                        G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                            if (G.core[w] < G.core[v]) {
                                ucr_values[w].r++;
                            }
                        });
                    } else {
                        // 传统方式
                        for (vertex_t w : G.adj.neighbors(v)) {
                            if (G.core[w] < G.core[v]) {
                                ucr_values[w].r++;
                            }
                        }
                    }
                }
                
                // 增量更新s值，仅处理可能受影响的顶点
                std::unordered_set<vertex_t> update_set;
                for (vertex_t v : all_upgraded) {
                    update_set.insert(v);
                    if (use_sentinel) {
                        // 使用哨兵模式优化
                        G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                            update_set.insert(w);
                            G.adj.for_each_neighbor_with_sentinel(w, [&](vertex_t z) {
                                if (G.core[z] == G.core[w]) {
                                    update_set.insert(z);
                                }
                            });
                        });
                    } else {
                        // 传统方式
                        for (vertex_t w : G.adj.neighbors(v)) {
                            update_set.insert(w);
                            for (vertex_t z : G.adj.neighbors(w)) {
                                if (G.core[z] == G.core[w]) {
                                    update_set.insert(z);
                                }
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
                        
                        if (use_sentinel) {
                            // 使用哨兵模式优化
                            G.adj.for_each_neighbor_with_sentinel(u, [&](vertex_t w) {
                                if (G.core[w] == k && !visited.count(w) && candidates.count(w)) {
                                    visited.insert(w);
                                    q.push(w);
                                }
                            });
                        } else {
                            // 传统方式
                            for (vertex_t w : G.adj.neighbors(u)) {
                                if (G.core[w] == k && !visited.count(w) && candidates.count(w)) {
                                    visited.insert(w);
                                    q.push(w);
                                }
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
                
                if (use_sentinel) {
                    // 使用哨兵模式优化
                    G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                        if (connected_core.count(w) && !evicted.count(w) && 
                            G.core[w] == k && is_qualified(v)) {
                            cd[w]--;
                        }
                    });
                } else {
                    // 传统方式
                    for (vertex_t w : G.adj.neighbors(v)) {
                        if (connected_core.count(w) && !evicted.count(w) && 
                            G.core[w] == k && is_qualified(v)) {
                            cd[w]--;
                        }
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
                    if (use_sentinel) {
                        // 使用哨兵模式优化
                        G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                            if (G.core[w] == k && !processed.count(w)) {
                                next_frontier.insert(w);
                            }
                        });
                    } else {
                        // 传统方式
                        for (vertex_t w : G.adj.neighbors(v)) {
                            if (G.core[w] == k && !processed.count(w)) {
                                next_frontier.insert(w);
                            }
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
                   
                   if (use_sentinel) {
                       // 使用哨兵模式优化
                       G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                           if (G.core[w] == k && !visited.count(w)) {
                               visited.insert(w);
                               q.push(w);
                           }
                       });
                   } else {
                       // 传统方式
                       for (vertex_t w : G.adj.neighbors(v)) {
                           if (G.core[w] == k && !visited.count(w)) {
                               visited.insert(w);
                               q.push(w);
                           }
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
           if (use_sentinel) {
               // 使用哨兵模式优化
               G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                   if (G.core[w] >= k) count++;
               });
           } else {
               // 传统方式
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (G.core[w] >= k) count++;
               }
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
           if (use_sentinel) {
               // 使用哨兵模式优化
               G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                   if (G.core[w] == k && !degraded.count(w)) {
                       eff_deg[w]--;
                       if (eff_deg[w] < k) {
                           degrade_queue.push_back(w);
                       }
                   }
               });
           } else {
               // 传统方式
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (G.core[w] == k && !degraded.count(w)) {
                       eff_deg[w]--;
                       if (eff_deg[w] < k) {
                           degrade_queue.push_back(w);
                       }
                   }
               }
           }
       }
       
       // 更新被降级顶点的邻居的r值
       for (vertex_t v : degraded) {
           if (use_sentinel) {
               // 使用哨兵模式优化
               G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                   if (G.core[w] > k - 1) {
                       // 顶点w的邻居v从k降到k-1
                       ucr_values[w].r--;
                   }
               });
           } else {
               // 传统方式
               for (vertex_t w : G.adj.neighbors(v)) {
                   if (G.core[w] > k - 1) {
                       // 顶点w的邻居v从k降到k-1
                       ucr_values[w].r--;
                   }
               }
           }
       }
       
       // 增量更新s值
       std::unordered_set<vertex_t> update_set;
       for (vertex_t v : degraded) {
           update_set.insert(v);
           if (use_sentinel) {
               // 使用哨兵模式优化
               G.adj.for_each_neighbor_with_sentinel(v, [&](vertex_t w) {
                   update_set.insert(w);
               });
           } else {
               // 传统方式
               for (vertex_t w : G.adj.neighbors(v)) {
                   update_set.insert(w);
               }
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

// 实验运行器 - 用于测试不同线程数和哨兵优化
class ThreadSentinelExperimentRunner {
private:
    TemporalGraphProcessor& processor;
    int window_size;
    int starting_bin;
    int num_steps;
    std::vector<int> thread_counts;
    
public:
    ThreadSentinelExperimentRunner(TemporalGraphProcessor& proc, int win_size, int start_bin, int steps, 
                                  const std::vector<int>& threads)
        : processor(proc), window_size(win_size), starting_bin(start_bin), num_steps(steps), thread_counts(threads) {}
    
    void run() {
        struct ExperimentResult {
            int threads;
            bool use_sentinel;
            double init_time;
            double total_update_time;
            double avg_update_time;
            size_t memory_usage;
        };
        
        std::vector<ExperimentResult> results;
        
        // 测试不同线程数和有无哨兵优化的组合
        for (int threads : thread_counts) {
            for (int sentinel_mode = 0; sentinel_mode <= 1; ++sentinel_mode) {
                bool use_sentinel = (sentinel_mode == 1);
                std::string mode_str = use_sentinel ? "启用哨兵模式" : "不使用哨兵";
                
                std::cout << "\n=========================================================" << std::endl;
                std::cout << "运行实验: 线程数 = " << threads << ", " << mode_str << std::endl;
                std::cout << "=========================================================\n" << std::endl;
                
                // 设置OpenMP线程数
                omp_set_num_threads(threads);
                std::cout << "使用 " << threads << " 个OpenMP线程" << std::endl;
                
                // 初始化图
                Graph graph;
                
                // 获取所有bins
                size_t num_bins = processor.get_num_bins();
                if (starting_bin + window_size > num_bins) {
                    std::cerr << "无效的起始bin或窗口大小!" << std::endl;
                    continue;
                }
                
                // 构建初始窗口
                Timer window_timer;
                std::cout << "构建初始窗口 (bins " << starting_bin << " 到 " 
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
                std::cout << "添加了 " << all_edges.size() << " 条边，耗时 " << add_edges_timer.elapsed_seconds() << " 秒" << std::endl;
                
                // 计算初始核心度
                Timer core_timer;
                graph.compute_core_numbers();
                std::cout << "计算初始核心度，耗时 " << core_timer.elapsed_seconds() << " 秒" << std::endl;
                
                // 初始化UCR方法的状态
                Timer reset_timer;
                OptimizedUCRCore ucr_core(graph, use_sentinel);
                ucr_core.reset(true);
                double init_time = reset_timer.elapsed_seconds();
                std::cout << "初始化UCR方法，耗时 " << init_time << " 秒" << std::endl;
                
                // 估计初始内存使用
                size_t ucr_memory = ucr_core.estimate_memory_usage();
                
                std::cout << "初始窗口构建完成，耗时 " << window_timer.elapsed_seconds() << " 秒. 总共边数: " << all_edges.size() << std::endl;
                std::cout << "UCR内存使用: " << (ucr_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
                
                // 开始滑动窗口实验
                std::cout << "\n开始滑动窗口实验..." << std::endl;
                
                std::vector<double> update_times;
                double total_update_time = 0.0;
                
                for (int step = 0; step < num_steps; ++step) {
                    int remove_bin = starting_bin + step;
                    int add_bin = starting_bin + window_size + step;
                    
                    if (add_bin >= num_bins) break;
                    
                    std::cout << "\n步骤 " << step + 1 << "/" << num_steps 
                              << ": 移除bin " << remove_bin << ", 添加bin " << add_bin << std::endl;
                    
                    // 获取要删除和添加的边
                    const auto& remove_edges = processor.get_bin_edges(remove_bin);
                    const auto& add_edges = processor.get_bin_edges(add_bin);
                    
                    std::cout << "删除 " << remove_edges.size() << " 条边, 添加 " << add_edges.size() << " 条边" << std::endl;
                    
                    // 测试UCR方法
                    Timer update_timer;
                    double ucr_remove_time = ucr_core.batch_remove(remove_edges);
                    double ucr_add_time = ucr_core.batch_insert(add_edges);
                    double total_time = update_timer.elapsed_milliseconds();
                    
                    std::cout << "UCR删除时间: " << ucr_remove_time << " 毫秒" << std::endl;
                    std::cout << "UCR添加时间: " << ucr_add_time << " 毫秒" << std::endl;
                    std::cout << "总更新时间: " << total_time << " 毫秒" << std::endl;
                    
                    total_update_time += total_time;
                    update_times.push_back(total_time);
                }
                
                // 计算平均更新时间
                double avg_update_time = total_update_time / update_times.size();
                
                std::cout << "\n" << std::string(50, '=') << std::endl;
                std::cout << "实验结果: 线程数 = " << threads << ", " << mode_str << std::endl;
                std::cout << "初始化时间: " << init_time << " 秒" << std::endl;
                std::cout << "总更新时间: " << total_update_time << " 毫秒" << std::endl;
                std::cout << "平均每步更新时间: " << avg_update_time << " 毫秒" << std::endl;
                std::cout << "内存使用: " << (ucr_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
                std::cout << std::string(50, '=') << "\n" << std::endl;
                
                // 保存结果
                results.push_back({
                    threads,
                    use_sentinel,
                    init_time,
                    total_update_time,
                    avg_update_time,
                    ucr_memory
                });
            }
        }
        
        // 汇总比较不同设置的结果
        std::cout << "\n========== 实验结果汇总 ==========\n" << std::endl;
        std::cout << "线程数,哨兵模式,初始化时间(秒),总更新时间(毫秒),平均每步更新时间(毫秒),内存使用(MB)" << std::endl;
        
        for (const auto& result : results) {
            std::cout << result.threads << ","
                     << (result.use_sentinel ? "启用" : "禁用") << ","
                     << result.init_time << ","
                     << result.total_update_time << ","
                     << result.avg_update_time << ","
                     << (result.memory_usage / (1024.0 * 1024.0)) << std::endl;
        }
        
        // 对比各配置性能
        std::cout << "\n========== 性能对比 ==========\n" << std::endl;
        
        // 以单线程无哨兵模式为基准进行对比
        double baseline_init_time = 0;
        double baseline_update_time = 0;
        
        for (const auto& result : results) {
            if (result.threads == thread_counts[0] && !result.use_sentinel) {
                baseline_init_time = result.init_time;
                baseline_update_time = result.avg_update_time;
                break;
            }
        }
        
        if (baseline_init_time > 0 && baseline_update_time > 0) {
            std::cout << "配置,初始化加速比,更新加速比" << std::endl;
            
            for (const auto& result : results) {
                if (result.threads == thread_counts[0] && !result.use_sentinel) continue; // 跳过基准配置
                
                double init_speedup = baseline_init_time / result.init_time;
                double update_speedup = baseline_update_time / result.avg_update_time;
                
                std::cout << result.threads << "线程,"
                         << (result.use_sentinel ? "启用哨兵" : "禁用哨兵") << ","
                         << init_speedup << "x,"
                         << update_speedup << "x" << std::endl;
            }
        }
        
        // 将结果保存到CSV文件
        std::string csv_filename = "thread_sentinel_experiment_results.csv";
        std::ofstream csv_file(csv_filename);
        
        if (csv_file.is_open()) {
            csv_file << "线程数,哨兵模式,初始化时间(秒),总更新时间(毫秒),平均每步更新时间(毫秒),内存使用(MB),初始化加速比,更新加速比" << std::endl;
            
            for (const auto& result : results) {
                double init_speedup = baseline_init_time > 0 ? baseline_init_time / result.init_time : 1.0;
                double update_speedup = baseline_update_time > 0 ? baseline_update_time / result.avg_update_time : 1.0;
                
                csv_file << result.threads << ","
                        << (result.use_sentinel ? "启用" : "禁用") << ","
                        << result.init_time << ","
                        << result.total_update_time << ","
                        << result.avg_update_time << ","
                        << (result.memory_usage / (1024.0 * 1024.0)) << ","
                        << init_speedup << ","
                        << update_speedup << std::endl;
            }
            
            csv_file.close();
            std::cout << "\n结果已保存到 " << csv_filename << std::endl;
        } else {
            std::cerr << "无法打开文件进行写入: " << csv_filename << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    // 默认参数
    std::string filepath = "sx-stackoverflow.txt";
    int num_bins = 1000;
    int starting_bin = 400;
    int num_steps = 10;
    int window_size = 100;
    
    // 解析命令行参数
    if (argc > 1) filepath = argv[1];
    if (argc > 2) num_bins = std::stoi(argv[2]);
    if (argc > 3) starting_bin = std::stoi(argv[3]);
    if (argc > 4) num_steps = std::stoi(argv[4]);
    if (argc > 5) window_size = std::stoi(argv[5]);
    
    std::cout << "配置:" << std::endl;
    std::cout << "- 数据集: " << filepath << std::endl;
    std::cout << "- Bin数量: " << num_bins << std::endl;
    std::cout << "- 起始bin: " << starting_bin << std::endl;
    std::cout << "- 步骤数量: " << num_steps << std::endl;
    std::cout << "- 窗口大小: " << window_size << std::endl;
    
    // 加载时序图
    Timer load_timer;
    std::cout << "加载时序图数据..." << std::endl;
    TemporalGraphProcessor processor(filepath);
    std::cout << "数据加载完成，耗时 " << load_timer.elapsed_seconds() << " 秒." << std::endl;
    
    // 将数据分成指定数量的bins
    Timer bin_timer;
    std::cout << "创建 " << num_bins << " 个bins..." << std::endl;
    processor.create_bins(num_bins);
    std::cout << "Bins创建完成，耗时 " << bin_timer.elapsed_seconds() << " 秒." << std::endl;
    
    // 要测试的线程数
    std::vector<int> thread_counts = {4, 8, 16, 32};
    
    // 创建并运行线程和哨兵优化实验
    std::cout << "开始线程数和哨兵模式对比实验..." << std::endl;
    Timer experiment_timer;
    ThreadSentinelExperimentRunner experiment(processor, window_size, starting_bin, num_steps, thread_counts);
    
    // 运行实验
    experiment.run();
    std::cout << "实验完成，总耗时 " << experiment_timer.elapsed_seconds() << " 秒." << std::endl;
    
    return 0;
}