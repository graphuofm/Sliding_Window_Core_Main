#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <string>
#include <memory>
#include <map>
#include <deque>
#include <sstream>
#include <stack>
#include <cmath>


using namespace std;
using namespace std::chrono;

// 定义顶点和时间戳类型
using vertex_t = int;
using timestamp_t = long long;
using duration_t = microseconds;

// 输入边的数据结构
struct EdgeRaw {
    vertex_t u, v;
    timestamp_t ts;
};

// 图数据结构
struct Graph {
    // 邻接表，adj[i]包含顶点i的所有邻居
    vector<vector<vertex_t>> adj;
    // 顶点的核数(coreness)
    vector<int> core;
    
    // 构造函数，初始化一个n个顶点的空图
    explicit Graph(size_t n = 0) : adj(n), core(n, 0) {}
    
    // 创建图的深拷贝
    Graph clone() const {
        Graph g(adj.size());
        g.adj = adj;
        g.core = core;
        return g;
    }
    
    // 确保图有足够空间容纳指定ID的顶点
    void ensure(size_t id) {
        if (adj.size() <= id) {
            adj.resize(id + 1);
            core.resize(id + 1, 0);
        }
    }
    
    // 添加一条边(u,v)到图中
    void addEdge(vertex_t u, vertex_t v) {
        // 自环(self-loop)不考虑，直接忽略
        if (u == v) {
            return;
        }
        
        // 确保图能容纳这两个顶点
        ensure(std::max(u, v));
        
        auto& lu = adj[u];
        auto& lv = adj[v];
        
        // 检查边是否已存在，避免重复添加
        if (std::find(lu.begin(), lu.end(), v) == lu.end()) {
            lu.push_back(v);
        }
        
        if (std::find(lv.begin(), lv.end(), u) == lv.end()) {
            lv.push_back(u);
        }
    }
    
    // 从图中删除一条边(u,v)
    void removeEdge(vertex_t u, vertex_t v) {
        // 检查顶点是否存在
        if (u >= adj.size() || v >= adj.size()) {
            return;
        }
        
        auto& lu = adj[u];
        auto& lv = adj[v];
        
        // 从邻接表中删除边
        lu.erase(std::remove(lu.begin(), lu.end(), v), lu.end());
        lv.erase(std::remove(lv.begin(), lv.end(), u), lv.end());
    }
    
    // 获取图中顶点数量
    size_t n() const {
        return adj.size();
    }
    
    // 获取图中边的数量
    size_t m() const {
        size_t edgeCount = 0;
        for (const auto& edges : adj) {
            edgeCount += edges.size();
        }
        return edgeCount / 2; // 因为无向图中每条边计算了两次
    }
    
    // 验证核数是否正确，返回是否有错误
    bool validateCore() const {
        bool hasError = false;
        
        // 对于每个顶点，检查其是否符合k-core定义
        for (size_t i = 0; i < n(); ++i) {
            if (core[i] > 0) { // 只检查核数大于0的顶点
                int k = core[i];
                int count = 0;
                
                // 统计核数≥k的邻居数量
                for (vertex_t v : adj[i]) {
                    if (core[v] >= k) {
                        count++;
                    }
                }
                
                // 如果邻居数小于核数，则存在错误
                if (count < k) {
                    std::cerr << "核数验证错误: 顶点 " << i << " 核数=" << k 
                              << " 但只有 " << count << " 个满足条件的邻居" << std::endl;
                    hasError = true;
                }
            }
        }
        
        return !hasError;
    }
};

// 实现Batagelj-Zaversnik算法计算k-core分解
vector<int> bz_core(const Graph& G) {
    int n = G.n();
    int md = 0;  // 最大度数
    
    vector<int> deg(n), vert(n), pos(n);
    
    // 第一步: 计算初始度数
    for (int v = 0; v < n; ++v) {
        deg[v] = G.adj[v].size();
        md = std::max(md, deg[v]);
    }
    
    // 第二步: 为每个度数创建桶(bin)
    vector<int> bin(md + 1, 0);
    for (int d : deg) {
        ++bin[d];
    }
    
    // 计算每个桶的起始位置
    int start = 0;
    for (int d = 0; d <= md; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }
    
    // 第三步: 将顶点放入对应的桶中
    for (int v = 0; v < n; ++v) {
        pos[v] = bin[deg[v]]++;
        vert[pos[v]] = v;
    }
    
    // 重置桶位置，指向每个度数的第一个顶点
    for (int d = md; d >= 1; --d) {
        bin[d] = bin[d - 1];
    }
    bin[0] = 0;
    
    // 第四步: 按照度数从小到大处理顶点
    for (int i = 0; i < n; ++i) {
        int v = vert[i];  // 当前处理的顶点
        
        // 更新v的所有邻居
        for (int u : G.adj[v]) {
            // 只处理度数大于当前顶点的邻居
            if (deg[u] > deg[v]) {
                // 获取邻居u当前的位置和对应度数的第一个顶点
                int du = deg[u], pu = pos[u];
                int pw = bin[du], w = vert[pw];
                
                // 如果u不是该度数的第一个顶点，交换位置
                if (u != w) {
                    std::swap(vert[pu], vert[pw]);
                    pos[u] = pw;
                    pos[w] = pu;
                }
                
                // 更新桶位置并减少u的度数
                ++bin[du];
                --deg[u];
            }
        }
    }
    
    return deg;  // 返回计算出的核数
}

// 度量结构，用于MCD和PCD计算
template<typename T>
struct ArrayMetric {
    vector<T> values;
    T def;
    
    explicit ArrayMetric(T d = 0, size_t size = 0) : values(size, d), def(d) {}
    
    void ensure(size_t id) {
        if (values.size() <= id) {
            values.resize(id + 1, def);
        }
    }
    
    T get(int v) const {
        return v < values.size() ? values[v] : def;
    }
    
    T& ref(int v) {
        ensure(v);
        return values[v];
    }
    
    void set(int v, T val) {
        ensure(v);
        values[v] = val;
    }
    
    void clear() {
        std::fill(values.begin(), values.end(), def);
    }
    
    void resize(size_t size) {
        values.resize(size, def);
    }
};

// 原子批处理增量K-Core维护类
class AtomicBatchIncCore {
protected:
    Graph& G;
    ArrayMetric<int> MCD{0}, PCD{0};
    
public:
    AtomicBatchIncCore(Graph& g) : G(g) {
        MCD.resize(G.n());
        PCD.resize(G.n());
        
        G.core = bz_core(G);
        
        rebuild();
    }
    
    bool qual(int x, int y) const {
        int k = G.core[x];
        return G.core[y] > k || (G.core[y] == k && MCD.get(y) > k);
    }
    
    duration_t rebuild() {
        auto start = high_resolution_clock::now();
        
        MCD.clear();
        PCD.clear();
        
        int n = G.n();
        
        for (int v = 0; v < n; ++v) {
            int c = 0;
            for (int w : G.adj[v]) {
                if (G.core[w] >= G.core[v]) {
                    ++c;
                }
            }
            MCD.set(v, c);
        }
        
        for (int v = 0; v < n; ++v) {
            int c = 0;
            for (int w : G.adj[v]) {
                if (qual(v, w)) {
                    ++c;
                }
            }
            PCD.set(v, c);
        }
        
        auto end = high_resolution_clock::now();
        return duration_cast<duration_t>(end - start);
    }
    
    void addLocal(int x, int y) {
        if (G.core[y] >= G.core[x]) {
            ++MCD.ref(x);
        }
        
        if (qual(x, y)) {
            ++PCD.ref(x);
        }
    }
    
    void delLocal(int x, int y) {
        if (G.core[y] >= G.core[x]) {
            --MCD.ref(x);
        }
        
        if (qual(x, y)) {
            --PCD.ref(x);
        }
    }
    
    // 批量插入边
    duration_t batchInsert(const vector<pair<vertex_t, vertex_t>>& edges) {
        auto totalStart = high_resolution_clock::now();
        
        // 添加所有边
        for (const auto& [u, v] : edges) {
            G.addEdge(u, v);
            
            size_t need = std::max(u, v) + 1;
            if (G.core.size() < need) {
                G.core.resize(need, 0);
            }
        }
        
        // 更新局部度量
        for (const auto& [u, v] : edges) {
            addLocal(u, v);
            addLocal(v, u);
        }
        
        // 找出可能需要升级的顶点
        unordered_set<int> candidates;
        
        for (const auto& [u, v] : edges) {
            int ku = G.core[u], kv = G.core[v];
            
            if (ku > kv) {
                if (PCD.get(v) > kv) candidates.insert(v);
            } else if (ku < kv) {
                if (PCD.get(u) > ku) candidates.insert(u);
            } else {
                if (PCD.get(u) > ku) candidates.insert(u);
                if (PCD.get(v) > kv) candidates.insert(v);
            }
        }
        
        // 按k值分组
        map<int, vector<int>> kGroups;
        for (int v : candidates) {
            kGroups[G.core[v]].push_back(v);
        }
        
        // 处理每个k值组
        for (auto& [k, verts] : kGroups) {
            unordered_set<int> processed;
            
            for (int v : verts) {
                if (processed.count(v) || G.core[v] != k) continue;
                
                // 构建子核
                vector<int> subcore;
                unordered_set<int> visited;
                queue<int> q;
                q.push(v);
                visited.insert(v);
                
                while (!q.empty()) {
                    int x = q.front();
                    q.pop();
                    
                    if (G.core[x] == k) {
                        subcore.push_back(x);
                        
                        for (int w : G.adj[x]) {
                            if (G.core[w] == k && !visited.count(w)) {
                                visited.insert(w);
                                q.push(w);
                            }
                        }
                    }
                }
                
                for (int x : subcore) {
                    processed.insert(x);
                }
                
                // 剥离过程
                unordered_set<int> evicted;
                unordered_map<int, int> cd;
                
                for (int x : subcore) {
                    cd[x] = PCD.get(x);
                }
                
                bool stable = false;
                while (!stable) {
                    stable = true;
                    vector<int> to_evict;
                    
                    for (int x : subcore) {
                        if (evicted.count(x)) continue;
                        
                        if (cd[x] <= k) {
                            to_evict.push_back(x);
                            evicted.insert(x);
                            stable = false;
                        }
                    }
                    
                    for (int x : to_evict) {
                        for (int w : G.adj[x]) {
                            if (find(subcore.begin(), subcore.end(), w) != subcore.end() &&
                                !evicted.count(w) && G.core[w] == k && qual(w, x)) {
                                cd[w]--;
                            }
                        }
                    }
                }
                
                // 升级未被剥离的顶点
                for (int x : subcore) {
                    if (!evicted.count(x)) {
                        ++G.core[x];
                    }
                }
            }
        }
        
        // 重建度量
        rebuild();
        
        auto totalEnd = high_resolution_clock::now();
        return duration_cast<duration_t>(totalEnd - totalStart);
    }
    
    // 批量删除边
    duration_t batchRemove(const vector<pair<vertex_t, vertex_t>>& edges) {
        auto totalStart = high_resolution_clock::now();
        
        // 更新局部度量
        for (const auto& [u, v] : edges) {
            if (u >= G.n() || v >= G.n()) {
                continue;
            }
            
            delLocal(u, v);
            delLocal(v, u);
        }
        
        // 按k值分组待处理顶点
        map<int, vector<int>> kCandidates;
        
        unordered_set<int> affectedVertices;
        for (const auto& [u, v] : edges) {
            if (u >= G.n() || v >= G.n()) continue;
            
            affectedVertices.insert(u);
            affectedVertices.insert(v);
        }
        
        for (int v : affectedVertices) {
            if (G.core[v] > 0) {
                kCandidates[G.core[v]].push_back(v);
            }
        }
        
        // 从图中删除所有边
        for (const auto& [u, v] : edges) {
            G.removeEdge(u, v);
        }
        
        // 按k值从小到大处理
        for (const auto& [k, seeds] : kCandidates) {
            vector<int> candidates;
            unordered_map<int, int> deg;
            unordered_set<int> processed;
            
            // 处理同一k值的所有种子顶点
            for (int seed : seeds) {
                if (processed.count(seed) || G.core[seed] != k) continue;
                
                queue<int> bfsQ;
                bfsQ.push(seed);
                processed.insert(seed);
                
                while (!bfsQ.empty()) {
                    int v = bfsQ.front();
                    bfsQ.pop();
                    
                    if (G.core[v] == k) {
                        candidates.push_back(v);
                        
                        // 计算有效度数
                        int eff_deg = 0;
                        for (int w : G.adj[v]) {
                            if (G.core[w] >= k) eff_deg++;
                        }
                        deg[v] = eff_deg;
                        
                        // 继续BFS
                        for (int w : G.adj[v]) {
                            if (G.core[w] == k && !processed.count(w)) {
                                processed.insert(w);
                                bfsQ.push(w);
                            }
                        }
                    }
                }
            }
            
            // 初始化级联队列
            queue<int> cascadeQ;
            for (int v : candidates) {
                if (deg[v] < k) {
                    cascadeQ.push(v);
                }
            }
            
            if (cascadeQ.empty()) continue;
            
            // 级联降核处理
            while (!cascadeQ.empty()) {
                int v = cascadeQ.front();
                cascadeQ.pop();
                
                if (G.core[v] != k) continue;
                
                --G.core[v];
                
                for (int w : G.adj[v]) {
                    if (G.core[w] == k) {
                        --deg[w];
                        if (deg[w] < k) {
                            cascadeQ.push(w);
                        }
                    }
                }
            }
        }
        
        // 重建度量
        rebuild();
        
        auto totalEnd = high_resolution_clock::now();
        return duration_cast<duration_t>(totalEnd - totalStart);
    }
};

// 五种选择器实现
class BaseSelector {
public:
    virtual ~BaseSelector() = default;
    virtual bool shouldUseIncremental(const Graph& G, const vector<pair<vertex_t, vertex_t>>& edges) = 0;
    virtual string getName() const = 0;
    virtual void updateHistory(bool usedIncremental, bool incrementalFaster) {}
};

// 1. 简单比例选择器
class SimpleRatioSelector : public BaseSelector {
public:
    bool shouldUseIncremental(const Graph& G, const vector<pair<vertex_t, vertex_t>>& edges) override {
        return edges.size() < 0.05 * G.m();
    }
    
    string getName() const override {
        return "简单比例选择器";
    }
};

// 2. 图密度感知选择器
class DensityAwareSelector : public BaseSelector {
public:
    bool shouldUseIncremental(const Graph& G, const vector<pair<vertex_t, vertex_t>>& edges) override {
        // 计算图密度: |E|/|V|
        double avgDegree = (double)G.m() * 2 / G.n();
        
        // 根据图密度调整阈值
        double threshold = 0.05; // 基础阈值
        if (avgDegree > 10) {
            // 密集图降低阈值，更倾向于重新计算
            threshold = 0.02;
        } else if (avgDegree < 5) {
            // 稀疏图提高阈值，更倾向于增量更新
            threshold = 0.1;
        }
        
        return edges.size() < threshold * G.m();
    }
    
    string getName() const override {
        return "图密度感知选择器";
    }
};

// 3. 边局部性选择器
class LocalitySelector : public BaseSelector {
public:
    bool shouldUseIncremental(const Graph& G, const vector<pair<vertex_t, vertex_t>>& edges) override {
        // 计算受影响的顶点数
        unordered_set<vertex_t> affectedVertices;
        for (const auto& [u, v] : edges) {
            affectedVertices.insert(u);
            affectedVertices.insert(v);
        }
        
        // 如果受影响的顶点少于总顶点的10%或边数少于总边的3%，使用增量更新
        return (affectedVertices.size() < 0.1 * G.n()) || (edges.size() < 0.03 * G.m());
    }
    
    string getName() const override {
        return "边局部性选择器";
    }
};

// 4. 历史表现选择器
class HistoryBasedSelector : public BaseSelector {
private:
    // 记录最近10次更新的性能数据
    deque<pair<bool, bool>> history; // (使用增量?, 增量更快?)
    
public:
    bool shouldUseIncremental(const Graph& G, const vector<pair<vertex_t, vertex_t>>& edges) override {
        // 初始时使用简单比例
        if (history.empty()) {
            return edges.size() < 0.05 * G.m();
        }
        
        // 计算历史准确率
        int correctCount = 0;
        for (const auto& [usedIncremental, incrementalFaster] : history) {
            if (usedIncremental == incrementalFaster) {
                correctCount++;
            }
        }
        
        double accuracy = (double)correctCount / history.size();
        
        // 如果历史决策准确率低于60%，切换策略
        bool decision = edges.size() < 0.05 * G.m();
        if (accuracy < 0.6) {
            decision = !decision; // 反转决策
        }
        
        return decision;
    }
    
    void updateHistory(bool usedIncremental, bool incrementalFaster) override {
        history.push_back({usedIncremental, incrementalFaster});
        if (history.size() > 10) {
            history.pop_front();
        }
    }
    
    string getName() const override {
        return "历史表现选择器";
    }
};

// 5. 综合选择器
class ComprehensiveSelector : public BaseSelector {
public:
    bool shouldUseIncremental(const Graph& G, const vector<pair<vertex_t, vertex_t>>& edges) override {
        int edgesToUpdate = edges.size();
        int totalEdges = G.m();
        int totalVertices = G.n();
        
        // 计算受影响的顶点数
        unordered_set<vertex_t> affectedVertices;
        for (const auto& [u, v] : edges) {
            affectedVertices.insert(u);
            affectedVertices.insert(v);
        }
        
        // 计算图密度
        double avgDegree = (double)totalEdges * 2 / totalVertices;
        
        // 1. 极端情况的快速决策
        if (edgesToUpdate < 10) return true;  // 非常少的更新，用增量
        if (edgesToUpdate > 0.3 * totalEdges) return false;  // 大量更新，用BZ
        
        // 2. 结合多种因素计算分数
        double score = 0;
        
        // 更新比例因子 (0-1分)
        double ratioFactor = (double)edgesToUpdate / totalEdges;
        score += (ratioFactor > 0.1) ? 1 : 0;
        
        // 密度因子 (0-1分)
        score += (avgDegree > 8) ? 1 : 0;
        
        // 局部性因子 (0-1分)
        double localityFactor = (double)affectedVertices.size() / totalVertices;
        score += (localityFactor > 0.2) ? 1 : 0;
        
        // 总分大于等于2分，选择BZ，否则选择增量
        return score < 2;
    }
    
    string getName() const override {
        return "综合选择器";
    }
};

// 实验评估结构
struct SelectorEvaluationMetrics {
    int correctDecisions = 0;
    int totalDecisions = 0;
    double totalTimeSaved = 0;  // 微秒
    double totalTimeWasted = 0;  // 微秒
    double totalIncrementalTime = 0;  // 微秒
    double totalBZTime = 0;  // 微秒
    
    void printSummary() const {
        if (totalDecisions == 0) return;
        
        double accuracy = 100.0 * correctDecisions / totalDecisions;
        
        cout << "准确率: " << fixed << setprecision(2) << accuracy << "% (" 
             << correctDecisions << "/" << totalDecisions << ")" << endl;
        cout << "总增量时间: " << totalIncrementalTime << " 微秒, 平均每次: "
             << (totalIncrementalTime / totalDecisions) << " 微秒" << endl;
        cout << "总BZ时间: " << totalBZTime << " 微秒, 平均每次: "
             << (totalBZTime / totalDecisions) << " 微秒" << endl;
        cout << "总节省时间: " << totalTimeSaved << " 微秒" << endl;
        cout << "总浪费时间: " << totalTimeWasted << " 微秒" << endl;
        cout << "净收益: " << (totalTimeSaved - totalTimeWasted) << " 微秒" << endl;
    }
};

// 读取时序图数据
vector<EdgeRaw> readTemporalGraph(const string& filename) {
    vector<EdgeRaw> edges;
    ifstream file(filename);
    if (!file) {
        cerr << "无法打开文件: " << filename << endl;
        return edges;
    }
    
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        istringstream iss(line);
        EdgeRaw edge;
        if (!(iss >> edge.u >> edge.v >> edge.ts)) continue;
        
        // 忽略自环
        if (edge.u != edge.v) {
            edges.push_back(edge);
        }
    }
    
    // 按时间戳排序
    sort(edges.begin(), edges.end(), [](const EdgeRaw& a, const EdgeRaw& b) {
        return a.ts < b.ts;
    });
    
    cout << "从文件 " << filename << " 读取了 " << edges.size() << " 条边" << endl;
    return edges;
}

// 边分组方法枚举
enum class BinningMethod {
    EDGE_COUNT,   // 按边数均分
    TIMESTAMP     // 按时间戳均分
};

// 将边分配到bins中
vector<vector<pair<vertex_t, vertex_t>>> assignEdgesToBins(
    const vector<EdgeRaw>& edges, 
    int numBins, 
    BinningMethod method) {
    
    vector<vector<pair<vertex_t, vertex_t>>> bins(numBins);
    
    if (edges.empty() || numBins <= 0) {
        return bins;
    }
    
    if (method == BinningMethod::EDGE_COUNT) {
        // 按边数均分
        int edgesPerBin = (edges.size() + numBins - 1) / numBins;  // 向上取整
        
        int binIndex = 0;
        for (const auto& edge : edges) {
            if (bins[binIndex].size() >= edgesPerBin && binIndex < numBins - 1) {
                binIndex++;
            }
            bins[binIndex].emplace_back(edge.u, edge.v);
        }
    } else {
        // 按时间戳均分
        timestamp_t minTs = edges.front().ts;
        timestamp_t maxTs = edges.back().ts;
        double tsRange = static_cast<double>(maxTs - minTs);
        
        for (const auto& edge : edges) {
            // 计算这条边应该属于哪个bin
            int binIndex = static_cast<int>((edge.ts - minTs) / tsRange * numBins);
            binIndex = min(binIndex, numBins - 1); // 确保不会越界
            
            bins[binIndex].emplace_back(edge.u, edge.v);
        }
    }
    
    // 输出每个bin的大小
    cout << "Bin分配结果:" << endl;
    int totalEdges = 0;
    for (int i = 0; i < bins.size(); ++i) {
        totalEdges += bins[i].size();
        if (i % 1000 == 0 || i == bins.size() - 1) {
            cout << "Bin " << i << " 大小: " << bins[i].size() << " 条边" << endl;
        }
    }
    cout << "总计分配了 " << totalEdges << " 条边到 " << numBins << " 个bins中" << endl;
    
    return bins;
}

// 滑动窗口实验
void runSlideWindowExperiment(
    const vector<EdgeRaw>& edgeData,
    int numBins,
    int windowSize,
    int numSlides,
    BinningMethod method,
    const string& methodName) {
    
    cout << "\n========== 开始 " << methodName << " 实验 ==========\n" << endl;
    cout << "配置: " << numBins << " bins, 窗口大小: " << windowSize << " bins, 滑动次数: " << numSlides << endl;
    
    // 分配边到bins
    auto bins = assignEdgesToBins(edgeData, numBins, method);
    
    // 确定起始位置（选择图的中部）
    int startPos = (numBins - windowSize) / 2;
    if (startPos < 0) startPos = 0;
    
    cout << "起始位置: Bin " << startPos << endl;
    
    // 初始化所有选择器
    vector<unique_ptr<BaseSelector>> selectors;
    selectors.push_back(make_unique<SimpleRatioSelector>());
    selectors.push_back(make_unique<DensityAwareSelector>());
    selectors.push_back(make_unique<LocalitySelector>());
    selectors.push_back(make_unique<HistoryBasedSelector>());
    selectors.push_back(make_unique<ComprehensiveSelector>());
    
    // 为每个选择器准备评估指标
    vector<SelectorEvaluationMetrics> metrics(selectors.size());
    
    // 构建初始窗口
    Graph g;
    for (int i = startPos; i < startPos + windowSize && i < bins.size(); ++i) {
        for (const auto& [u, v] : bins[i]) {
            g.addEdge(u, v);
        }
    }
    
    // 初始化k-core
    g.core = bz_core(g);
    
    cout << "初始窗口包含 " << g.m() << " 条边，" << g.n() << " 个顶点" << endl;
    
    // 滑动窗口
    for (int slide = 0; slide < numSlides && startPos + windowSize + slide < bins.size(); ++slide) {
        cout << "\n--- 滑动 #" << (slide + 1) << " ---" << endl;
        
        // 要移除的bin（窗口左侧）
        int removeIndex = startPos + slide;
        
        // 要添加的bin（窗口右侧）
        int addIndex = startPos + windowSize + slide;
        
        vector<pair<vertex_t, vertex_t>> edgesToRemove = bins[removeIndex];
        vector<pair<vertex_t, vertex_t>> edgesToAdd = bins[addIndex];
        
        cout << "移除Bin " << removeIndex << " (" << edgesToRemove.size() << "条边), " 
             << "添加Bin " << addIndex << " (" << edgesToAdd.size() << "条边)" << endl;
        
        // 保存当前图状态，用于重置
        Graph originalGraph = g.clone();
        
        // 对每个选择器评估性能
        for (size_t i = 0; i < selectors.size(); ++i) {
            auto& selector = selectors[i];
            auto& metric = metrics[i];
            
            // 评估移除边的选择
            Graph tempGraph = originalGraph.clone();
            bool useIncrementalForRemove = selector->shouldUseIncremental(tempGraph, edgesToRemove);
            
            // 测试增量更新和BZ重算的实际时间
            // 对于移除操作
            
            // 方法1: BZ重算
            tempGraph = originalGraph.clone();
            auto bzStartRemove = high_resolution_clock::now();
            for (const auto& [u, v] : edgesToRemove) {
                tempGraph.removeEdge(u, v);
            }
            tempGraph.core = bz_core(tempGraph);
            auto bzEndRemove = high_resolution_clock::now();
            auto bzTimeRemove = duration_cast<duration_t>(bzEndRemove - bzStartRemove);
            
            // 方法2: 增量更新
            tempGraph = originalGraph.clone();
            AtomicBatchIncCore incCoreRemove(tempGraph);
            auto incStartRemove = high_resolution_clock::now();
            auto incTimeRemove = incCoreRemove.batchRemove(edgesToRemove);
            auto incEndRemove = high_resolution_clock::now();
            
            // 更新图状态
            if (useIncrementalForRemove) {
                tempGraph = originalGraph.clone();
                AtomicBatchIncCore inc(tempGraph);
                inc.batchRemove(edgesToRemove);
                g = tempGraph.clone();
            } else {
                tempGraph = originalGraph.clone();
                for (const auto& [u, v] : edgesToRemove) {
                    tempGraph.removeEdge(u, v);
                }
                tempGraph.core = bz_core(tempGraph);
                g = tempGraph.clone();
            }
            
            // 评估增加边的选择
            originalGraph = g.clone();
            bool useIncrementalForAdd = selector->shouldUseIncremental(originalGraph, edgesToAdd);
            
            // 方法1: BZ重算
            tempGraph = originalGraph.clone();
            auto bzStartAdd = high_resolution_clock::now();
            for (const auto& [u, v] : edgesToAdd) {
                tempGraph.addEdge(u, v);
            }
            tempGraph.core = bz_core(tempGraph);
            auto bzEndAdd = high_resolution_clock::now();
            auto bzTimeAdd = duration_cast<duration_t>(bzEndAdd - bzStartAdd);
            
            // 方法2: 增量更新
            tempGraph = originalGraph.clone();
            AtomicBatchIncCore incCoreAdd(tempGraph);
            auto incStartAdd = high_resolution_clock::now();
            auto incTimeAdd = incCoreAdd.batchInsert(edgesToAdd);
            auto incEndAdd = high_resolution_clock::now();
            
            // 更新图状态
            if (useIncrementalForAdd) {
                tempGraph = originalGraph.clone();
                AtomicBatchIncCore inc(tempGraph);
                inc.batchInsert(edgesToAdd);
                g = tempGraph.clone();
            } else {
                tempGraph = originalGraph.clone();
                for (const auto& [u, v] : edgesToAdd) {
                    tempGraph.addEdge(u, v);
                }
                tempGraph.core = bz_core(tempGraph);
                g = tempGraph.clone();
            }
            
            // 评估移除边的选择是否正确
            bool incrementalFasterRemove = incTimeRemove.count() < bzTimeRemove.count();
            bool correctDecisionRemove = (useIncrementalForRemove == incrementalFasterRemove);
            
            // 评估添加边的选择是否正确
            bool incrementalFasterAdd = incTimeAdd.count() < bzTimeAdd.count();
            bool correctDecisionAdd = (useIncrementalForAdd == incrementalFasterAdd);
            
            // 更新选择器历史
            selector->updateHistory(useIncrementalForRemove, incrementalFasterRemove);
            selector->updateHistory(useIncrementalForAdd, incrementalFasterAdd);
            
            // 更新性能指标（移除边）
            metric.totalDecisions++;
            metric.totalIncrementalTime += incTimeRemove.count();
            metric.totalBZTime += bzTimeRemove.count();
            
            if (correctDecisionRemove) {
                metric.correctDecisions++;
                metric.totalTimeSaved += abs(incTimeRemove.count() - bzTimeRemove.count());
            } else {
                metric.totalTimeWasted += abs(incTimeRemove.count() - bzTimeRemove.count());
            }
            
            // 更新性能指标（添加边）
            metric.totalDecisions++;
            metric.totalIncrementalTime += incTimeAdd.count();
            metric.totalBZTime += bzTimeAdd.count();
            
            if (correctDecisionAdd) {
                metric.correctDecisions++;
                metric.totalTimeSaved += abs(incTimeAdd.count() - bzTimeAdd.count());
            } else {
                metric.totalTimeWasted += abs(incTimeAdd.count() - bzTimeAdd.count());
            }
            
            cout << "选择器 " << selector->getName() 
                 << " - 移除选择: " << (useIncrementalForRemove ? "增量" : "BZ") 
                 << (correctDecisionRemove ? " ✓" : " ✗")
                 << " (Inc: " << incTimeRemove.count() << "μs, BZ: " << bzTimeRemove.count() << "μs), "
                 << "添加选择: " << (useIncrementalForAdd ? "增量" : "BZ") 
                 << (correctDecisionAdd ? " ✓" : " ✗")
                 << " (Inc: " << incTimeAdd.count() << "μs, BZ: " << bzTimeAdd.count() << "μs)"
                 << endl;
        }
    }
    
    // 输出最终结果
    cout << "\n========== " << methodName << " 实验结果 ==========\n" << endl;
    
    for (size_t i = 0; i < selectors.size(); ++i) {
        cout << "--- 选择器: " << selectors[i]->getName() << " ---" << endl;
        metrics[i].printSummary();
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    // 默认文件路径
    string filePath = "sx-stackoverflow.txt";
    
    // 允许命令行参数修改文件路径
    if (argc > 1) {
        filePath = argv[1];
    }
    
    // 读取时序图数据
    auto edgeData = readTemporalGraph(filePath);
    
    if (edgeData.empty()) {
        cerr << "没有读取到数据，退出" << endl;
        return 1;
    }
    
    // 实验1: 按边数量划分为10000个bins，窗口大小为100bins
    runSlideWindowExperiment(
        edgeData,
        10000,
        100,
        10,
        BinningMethod::EDGE_COUNT,
        "实验1: 按边数量划分(10000 bins)"
    );
    
    // 实验2: 按边数量划分为100000个bins，窗口大小为1000bins
    runSlideWindowExperiment(
        edgeData,
        100000,
        1000,
        10,
        BinningMethod::EDGE_COUNT,
        "实验2: 按边数量划分(100000 bins)"
    );
    
    // 实验3: 按时间戳划分为10000个bins，窗口大小为100bins
    runSlideWindowExperiment(
        edgeData,
        10000,
        100,
        10,
        BinningMethod::TIMESTAMP,
        "实验3: 按时间戳划分(10000 bins)"
    );
    
    // 实验4: 按时间戳划分为100000个bins，窗口大小为1000bins
    runSlideWindowExperiment(
        edgeData,
        100000,
        1000,
        10,
        BinningMethod::TIMESTAMP,
        "实验4: 按时间戳划分(100000 bins)"
    );
    
    return 0;
}