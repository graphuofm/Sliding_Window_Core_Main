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
#include <memory>
#include <climits>
#include <cstdlib>

using namespace std;
using vertex_t = int;
using edge_t = pair<vertex_t, vertex_t>;
using timestamp_t = long long;

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

// 时序边结构
struct TemporalEdge {
    vertex_t src;
    vertex_t dst;
    timestamp_t timestamp;
};

// 图结构
struct Graph {
    vector<vector<vertex_t>> adj;
    vector<int> core;
    
    explicit Graph(size_t n = 0) : adj(n), core(n, 0) {}
    
    Graph clone() const {
        Graph g(adj.size());
        g.adj = adj;
        g.core = core;
        return g;
    }
    
    void ensure(size_t id) {
        if (adj.size() <= id) {
            adj.resize(id + 1);
            core.resize(id + 1, 0);
        }
    }
    
    void addEdge(vertex_t u, vertex_t v) {
        if (u == v) return;
        
        ensure(max(u, v));
        
        auto& lu = adj[u];
        auto& lv = adj[v];
        
        if (find(lu.begin(), lu.end(), v) == lu.end()) {
            lu.push_back(v);
        }
        
        if (find(lv.begin(), lv.end(), u) == lv.end()) {
            lv.push_back(u);
        }
    }
    
    void removeEdge(vertex_t u, vertex_t v) {
        if (u >= adj.size() || v >= adj.size()) return;
        
        auto& lu = adj[u];
        auto& lv = adj[v];
        
        lu.erase(remove(lu.begin(), lu.end(), v), lu.end());
        lv.erase(remove(lv.begin(), lv.end(), u), lv.end());
    }
    
    size_t n() const {
        return adj.size();
    }
};

// BZ算法计算核心度
vector<int> bz_core(const Graph& G) {
    int n = G.n();
    int md = 0;
    
    vector<int> deg(n), vert(n), pos(n);
    
    for (int v = 0; v < n; ++v) {
        deg[v] = G.adj[v].size();
        md = max(md, deg[v]);
    }
    
    vector<int> bin(md + 1, 0);
    for (int d : deg) {
        ++bin[d];
    }
    
    int start = 0;
    for (int d = 0; d <= md; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }
    
    for (int v = 0; v < n; ++v) {
        pos[v] = bin[deg[v]]++;
        vert[pos[v]] = v;
    }
    
    for (int d = md; d >= 1; --d) {
        bin[d] = bin[d - 1];
    }
    bin[0] = 0;
    
    for (int i = 0; i < n; ++i) {
        int v = vert[i];
        
        for (int u : G.adj[v]) {
            if (deg[u] > deg[v]) {
                int du = deg[u], pu = pos[u];
                int pw = bin[du], w = vert[pw];
                
                if (u != w) {
                    swap(vert[pu], vert[pw]);
                    pos[u] = pw;
                    pos[w] = pu;
                }
                
                ++bin[du];
                --deg[u];
            }
        }
    }
    
    return deg;
}

// UCR基础类
class BaseUCR {
protected:
    Graph& G;
    vector<int> r_values;  // r(v): neighbors with higher core number
    vector<int> s_values;  // s(v): eligible neighbors with equal core number
    
public:
    BaseUCR(Graph& g) : G(g) {
        r_values.resize(G.n());
        s_values.resize(G.n());
        
        G.core = bz_core(G);
        rebuild();
    }
    
    // 判断邻居是否eligible
    bool isEligible(vertex_t v) const {
        if (v >= r_values.size()) return false;
        int k = G.core[v];
        return r_values[v] + s_values[v] > k;
    }
    
    // 重建r和s值
    void rebuild() {
        r_values.clear();
        s_values.clear();
        r_values.resize(G.n(), 0);
        s_values.resize(G.n(), 0);
        
        int n = G.n();
        
        // 计算r值
        for (int v = 0; v < n; ++v) {
            int r_count = 0;
            for (int w : G.adj[v]) {
                if (G.core[w] > G.core[v]) {
                    ++r_count;
                }
            }
            r_values[v] = r_count;
        }
        
        // 计算s值
        for (int v = 0; v < n; ++v) {
            int s_count = 0;
            int k = G.core[v];
            
            for (int w : G.adj[v]) {
                if (G.core[w] == k && isEligible(w)) {
                    ++s_count;
                }
            }
            s_values[v] = s_count;
        }
    }
    
    // 局部更新r和s值（插入边时）
    void addLocal(vertex_t u, vertex_t v) {
        if (G.core[v] > G.core[u]) {
            ++r_values[u];
        }
        
        // 更新s值需要检查eligibility
        if (G.core[v] == G.core[u] && isEligible(v)) {
            ++s_values[u];
        }
    }
    
    // 局部更新r和s值（删除边时）
    void delLocal(vertex_t u, vertex_t v) {
        if (G.core[v] > G.core[u]) {
            --r_values[u];
        }
        
        if (G.core[v] == G.core[u] && isEligible(v)) {
            --s_values[u];
        }
    }
};

// UCR-Basic: 逐边处理
class UCRBasic : public BaseUCR {
public:
    using BaseUCR::BaseUCR;
    
    double processEdges(const vector<edge_t>& edges, bool isInsertion) {
        Timer timer;
        
        if (isInsertion) {
            for (const auto& [u, v] : edges) {
                processEdgeInsertion(u, v);
            }
        } else {
            for (const auto& [u, v] : edges) {
                processEdgeRemoval(u, v);
            }
        }
        
        rebuild();  // 最后重建r和s值
        return timer.elapsed_milliseconds();
    }
    
private:
    void processEdgeInsertion(vertex_t u, vertex_t v) {
        // 确保顶点存在
        size_t need = max(u, v) + 1;
        if (G.core.size() < need) {
            G.core.resize(need, 0);
            r_values.resize(need, 0);
            s_values.resize(need, 0);
        }
        
        G.addEdge(u, v);
        addLocal(u, v);
        addLocal(v, u);
        
        // 检查是否需要提升核心度
        if (G.core[u] > G.core[v] && r_values[v] + s_values[v] > G.core[v]) {
            promote(v);
        } else if (G.core[u] < G.core[v] && r_values[u] + s_values[u] > G.core[u]) {
            promote(u);
        } else if (G.core[u] == G.core[v]) {
            if (r_values[u] + s_values[u] > G.core[u]) promote(u);
            if (r_values[v] + s_values[v] > G.core[v]) promote(v);
        }
    }
    
    void processEdgeRemoval(vertex_t u, vertex_t v) {
        if (u >= G.n() || v >= G.n()) return;
        
        delLocal(u, v);
        delLocal(v, u);
        
        G.removeEdge(u, v);
        
        // 检查是否需要降低核心度
        if (r_values[u] + s_values[u] < G.core[u]) {
            degrade(u);
        }
        if (r_values[v] + s_values[v] < G.core[v]) {
            degrade(v);
        }
    }
    
    void promote(vertex_t root) {
        int k = G.core[root];
        
        if (r_values[root] + s_values[root] <= k) return;
        
        // 简化的提升过程
        queue<int> q;
        unordered_set<int> visited;
        
        q.push(root);
        visited.insert(root);
        
        vector<int> candidates;
        
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            
            candidates.push_back(v);
            
            for (int w : G.adj[v]) {
                if (G.core[w] == k && !visited.count(w)) {
                    visited.insert(w);
                    q.push(w);
                }
            }
        }
        
        // 简化处理：直接提升所有满足条件的顶点
        for (int v : candidates) {
            if (r_values[v] + s_values[v] > k) {
                ++G.core[v];
            }
        }
    }
    
    void degrade(vertex_t root) {
        int k = G.core[root];
        if (k == 0) return;
        
        queue<int> q;
        q.push(root);
        
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            
            if (G.core[v] != k) continue;
            
            --G.core[v];
            
            // 影响同层邻居
            for (int w : G.adj[v]) {
                if (G.core[w] == k) {
                    --r_values[w];  // v的核心度降低了
                    if (r_values[w] + s_values[w] < k) {
                        q.push(w);
                    }
                }
            }
        }
    }
};

// UCR-Batch: 批量处理（借鉴第一份代码的思想）
class UCRBatch : public BaseUCR {
public:
    using BaseUCR::BaseUCR;
    
    double processEdges(const vector<edge_t>& edges, bool isInsertion) {
        Timer timer;
        
        if (isInsertion) {
            batchInsert(edges);
        } else {
            batchRemove(edges);
        }
        
        rebuild();  // 最后重建r和s值
        return timer.elapsed_milliseconds();
    }
    
private:
    void batchInsert(const vector<edge_t>& edges) {
        // Phase 1: 批量更新图结构
        for (const auto& [u, v] : edges) {
            size_t need = max(u, v) + 1;
            if (G.core.size() < need) {
                G.core.resize(need, 0);
                r_values.resize(need, 0);
                s_values.resize(need, 0);
            }
            
            G.addEdge(u, v);
        }
        
        // Phase 2: 批量收集候选顶点
        unordered_set<int> candidateVertices;
        
        for (const auto& [u, v] : edges) {
            addLocal(u, v);
            addLocal(v, u);
            
            if (G.core[u] > G.core[v]) {
                if (r_values[v] + s_values[v] > G.core[v]) {
                    candidateVertices.insert(v);
                }
            } else if (G.core[u] < G.core[v]) {
                if (r_values[u] + s_values[u] > G.core[u]) {
                    candidateVertices.insert(u);
                }
            } else {
                if (r_values[u] + s_values[u] > G.core[u]) {
                    candidateVertices.insert(u);
                }
                if (r_values[v] + s_values[v] > G.core[v]) {
                    candidateVertices.insert(v);
                }
            }
        }
        
        // Phase 3: 批量处理所有候选（按k值分组）
        unordered_map<int, vector<int>> kGroups;
        for (int v : candidateVertices) {
            kGroups[G.core[v]].push_back(v);
        }
        
        // 从小到大处理每个k层
        vector<int> kValues;
        for (const auto& [k, _] : kGroups) {
            kValues.push_back(k);
        }
        sort(kValues.begin(), kValues.end());
        
        for (int k : kValues) {
            unordered_set<int> toPromote;
            
            // BFS收集k层的连通分量
            unordered_set<int> processed;
            
            for (int root : kGroups[k]) {
                if (processed.count(root)) continue;
                
                queue<int> q;
                unordered_set<int> component;
                
                q.push(root);
                component.insert(root);
                
                while (!q.empty()) {
                    int v = q.front();
                    q.pop();
                    
                    for (int w : G.adj[v]) {
                        if (G.core[w] == k && !component.count(w)) {
                            component.insert(w);
                            q.push(w);
                        }
                    }
                }
                
                // 批量评估这个连通分量
                for (int v : component) {
                    processed.insert(v);
                    if (r_values[v] + s_values[v] > k) {
                        toPromote.insert(v);
                    }
                }
            }
            
            // 批量提升核心度
            for (int v : toPromote) {
                ++G.core[v];
            }
        }
    }
    
    void batchRemove(const vector<edge_t>& edges) {
        // Phase 1: 批量更新局部度量
        for (const auto& [u, v] : edges) {
            if (u >= G.n() || v >= G.n()) continue;
            
            delLocal(u, v);
            delLocal(v, u);
        }
        
        // Phase 2: 批量删除边
        for (const auto& [u, v] : edges) {
            G.removeEdge(u, v);
        }
        
        // Phase 3: 批量处理降级（按k值分组）
        unordered_map<int, vector<int>> kGroups;
        
        for (const auto& [u, v] : edges) {
            if (u >= G.n() || v >= G.n()) continue;
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku > 0 && r_values[u] + s_values[u] < ku) {
                kGroups[ku].push_back(u);
            }
            if (kv > 0 && r_values[v] + s_values[v] < kv) {
                kGroups[kv].push_back(v);
            }
        }
        
        // 从大到小处理每个k层
        vector<int> kValues;
        for (const auto& [k, _] : kGroups) {
            kValues.push_back(k);
        }
        sort(kValues.rbegin(), kValues.rend());
        
        for (int k : kValues) {
            queue<int> cascadeQ;
            unordered_set<int> inQueue;
            
            for (int v : kGroups[k]) {
                if (G.core[v] == k && r_values[v] + s_values[v] < k) {
                    cascadeQ.push(v);
                    inQueue.insert(v);
                }
            }
            
            // 批量级联降级
            while (!cascadeQ.empty()) {
                int v = cascadeQ.front();
                cascadeQ.pop();
                
                if (G.core[v] != k) continue;
                
                --G.core[v];
                
                // 影响邻居
                for (int w : G.adj[v]) {
                    if (G.core[w] == k) {
                        --r_values[w];
                        if (r_values[w] + s_values[w] < k && !inQueue.count(w)) {
                            cascadeQ.push(w);
                            inQueue.insert(w);
                        }
                    }
                }
            }
        }
    }
};

// 实验主函数
void runExperiment(const string& dataset_name, const vector<TemporalEdge>& edges) {
    cout << "\n=====================================\n";
    cout << "Dataset: " << dataset_name << "\n";
    cout << "Total edges: " << edges.size() << "\n";
    cout << "=====================================\n";
    
    // 将边分配到bins
    const int total_bins = 100;
    timestamp_t min_ts = LLONG_MAX, max_ts = LLONG_MIN;
    
    for (const auto& e : edges) {
        min_ts = min(min_ts, e.timestamp);
        max_ts = max(max_ts, e.timestamp);
    }
    
    double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
    vector<vector<edge_t>> bins(total_bins);
    
    for (const auto& e : edges) {
        int bin_idx = min(static_cast<int>((e.timestamp - min_ts) / bin_span), total_bins - 1);
        bins[bin_idx].push_back({e.src, e.dst});
    }
    
    // 构建初始窗口
    int window_size = 10;
    int start_bin = 40;
    
    Graph initial_graph;
    for (int i = start_bin; i < start_bin + window_size && i < total_bins; ++i) {
        for (const auto& edge : bins[i]) {
            initial_graph.addEdge(edge.first, edge.second);
        }
    }
    
    cout << "Initial graph: " << initial_graph.adj.size() << " vertices\n";
    initial_graph.core = bz_core(initial_graph);
    
    // 测试不同的批次大小
    vector<int> batch_sizes = {1, 5, 10, 20, 50, 100};
    
    cout << "\nBatch_Size\tUCR-Basic(ms)\tUCR-Batch(ms)\tSpeedup\n";
    cout << "--------------------------------------------------------\n";
    
    for (int batch_size : batch_sizes) {
        // 准备测试边
        vector<edge_t> test_edges;
        int next_bin = start_bin + window_size;
        
        while (test_edges.size() < batch_size && next_bin < total_bins) {
            for (const auto& edge : bins[next_bin]) {
                test_edges.push_back(edge);
                if (test_edges.size() >= batch_size) break;
            }
            next_bin++;
        }
        
        if (test_edges.empty()) continue;
        
        // 测试UCR-Basic
        double basic_time;
        {
            Graph basic_graph = initial_graph.clone();
            UCRBasic ucr_basic(basic_graph);
            
            basic_time = ucr_basic.processEdges(test_edges, true);
        }
        
        // 测试UCR-Batch
        double batch_time;
        {
            Graph batch_graph = initial_graph.clone();
            UCRBatch ucr_batch(batch_graph);
            
            batch_time = ucr_batch.processEdges(test_edges, true);
        }
        
        double speedup = basic_time / batch_time;
        
        cout << batch_size << "\t\t" 
             << fixed << setprecision(3) << basic_time << "\t\t"
             << batch_time << "\t\t"
             << speedup << "x\n";
    }
}

int main() {
    cout << "===========================================\n";
    cout << "UCR Batch Processing Performance Analysis\n";
    cout << "===========================================\n";
    
    // 生成一个小规模的测试数据集
    vector<TemporalEdge> test_edges;
    
    // 创建一个具有社交网络特征的小图
    timestamp_t ts = 1;
    
    // 核心社区
    for (int i = 0; i < 50; ++i) {
        for (int j = i + 1; j < 50; ++j) {
            if (rand() % 100 < 10) {  // 10%概率连接
                test_edges.push_back({i, j, ts++});
            }
        }
    }
    
    // 外围社区
    for (int i = 50; i < 200; ++i) {
        // 连接到核心
        int core_node = rand() % 50;
        test_edges.push_back({i, core_node, ts++});
        
        // 外围之间的连接
        for (int j = i + 1; j < 200; ++j) {
            if (rand() % 100 < 2) {  // 2%概率
                test_edges.push_back({i, j, ts++});
            }
        }
    }
    
    // 动态添加的边
    for (int t = 0; t < 1000; ++t) {
        int u = rand() % 200;
        int v = rand() % 200;
        if (u != v) {
            test_edges.push_back({u, v, ts++});
        }
    }
    
    cout << "Generated synthetic dataset with " << test_edges.size() << " edges\n";
    
    runExperiment("synthetic_small", test_edges);
    
    // 如果需要，可以添加读取真实数据集的代码
    // 例如：sx-superuser 的一个子集
    
    cout << "\n===========================================\n";
    cout << "Experiment completed!\n";
    cout << "===========================================\n";
    
    return 0;
}