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
#include <random>
#include <set>

using vertex_t = int;
using timestamp_t = long long;
using time_point_t = std::chrono::high_resolution_clock::time_point;
using duration_t = std::chrono::microseconds;

struct EdgeRaw {
    vertex_t u, v;
    timestamp_t ts;
};

struct Graph {
    std::vector<std::vector<vertex_t>> adj;
    std::vector<int> core;
    
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
        if (u == v) {
            return;
        }
        
        ensure(std::max(u, v));
        
        auto& lu = adj[u];
        auto& lv = adj[v];
        
        if (std::find(lu.begin(), lu.end(), v) == lu.end()) {
            lu.push_back(v);
        }
        
        if (std::find(lv.begin(), lv.end(), u) == lv.end()) {
            lv.push_back(u);
        }
    }
    
    void removeEdge(vertex_t u, vertex_t v) {
        if (u >= adj.size() || v >= adj.size()) {
            return;
        }
        
        auto& lu = adj[u];
        auto& lv = adj[v];
        
        lu.erase(std::remove(lu.begin(), lu.end(), v), lu.end());
        lv.erase(std::remove(lv.begin(), lv.end(), u), lv.end());
    }
    
    size_t n() const {
        return adj.size();
    }
    
    size_t m() const {
        size_t count = 0;
        for (const auto& neighbors : adj) {
            count += neighbors.size();
        }
        return count / 2;
    }
    
    struct CoreDifference {
        int mismatchCount = 0;
        double avgDifference = 0.0;
        
        void print() const {
            std::cout << "不匹配顶点数: " << mismatchCount << std::endl;
            std::cout << "平均核数差异: " << avgDifference << std::endl;
        }
    };
    
    CoreDifference compareCores(const Graph& other) const {
        CoreDifference diff;
        
        size_t commonSize = std::min(core.size(), other.core.size());
        int totalDiff = 0;
        
        for (size_t i = 0; i < commonSize; ++i) {
            if (core[i] != other.core[i]) {
                diff.mismatchCount++;
                totalDiff += std::abs(core[i] - other.core[i]);
            }
        }
        
        if (diff.mismatchCount > 0) {
            diff.avgDifference = static_cast<double>(totalDiff) / diff.mismatchCount;
        }
        
        return diff;
    }
    
    bool validateCore() const {
        bool hasError = false;
        
        for (size_t i = 0; i < n(); ++i) {
            if (core[i] > 0) {
                int k = core[i];
                int count = 0;
                
                for (vertex_t v : adj[i]) {
                    if (core[v] >= k) {
                        count++;
                    }
                }
                
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

std::vector<int> bz_core(const Graph& G) {
    int n = G.n();
    int md = 0;
    
    std::vector<int> deg(n), vert(n), pos(n);
    
    for (int v = 0; v < n; ++v) {
        deg[v] = G.adj[v].size();
        md = std::max(md, deg[v]);
    }
    
    std::vector<int> bin(md + 1, 0);
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
                    std::swap(vert[pu], vert[pw]);
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

duration_t runBZAlgorithm(Graph& G) {
    auto start = std::chrono::high_resolution_clock::now();
    G.core = bz_core(G);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<duration_t>(end - start);
}

template<typename T>
struct ArrayMetric {
    std::vector<T> values;
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

template<typename T>
using Metric = ArrayMetric<T>;

struct BatchMetrics {
    duration_t totalTime{0};
    duration_t localUpdateTime{0};
    duration_t buildCandTime{0};
    duration_t cascadeTime{0};
    duration_t rebuildTime{0};
    duration_t graphUpdateTime{0};
    
    int edgeCount = 0;
    int batchCount = 0;
    
    size_t peakMemoryUsage = 0;
    
    void reset() {
        totalTime = duration_t{0};
        localUpdateTime = duration_t{0};
        buildCandTime = duration_t{0};
        cascadeTime = duration_t{0};
        rebuildTime = duration_t{0};
        graphUpdateTime = duration_t{0};
        edgeCount = 0;
        batchCount = 0;
        peakMemoryUsage = 0;
    }
    
    void print(const std::string& title) const {
        std::cout << "\n===== " << title << " (" << edgeCount << " 条边, " << batchCount << " 个批次) =====\n";
        
        if (edgeCount > 0) {
            double avgTotal = totalTime.count() / static_cast<double>(edgeCount);
            
            std::cout << "总时间: " << totalTime.count() << " 微秒, 平均: " 
                      << avgTotal << " 微秒/边\n";
            
            std::cout << "步骤细分 (总微秒 | 平均微秒/边 | 百分比):\n";
            printStep("图更新", graphUpdateTime, edgeCount, totalTime);
            printStep("局部更新", localUpdateTime, edgeCount, totalTime);
            printStep("构建候选集", buildCandTime, edgeCount, totalTime);
            printStep("级联处理", cascadeTime, edgeCount, totalTime);
            printStep("重建度量", rebuildTime, edgeCount, totalTime);
            
            std::cout << "峰值内存使用: " << (peakMemoryUsage / 1024.0 / 1024.0) << " MB\n";
        }
    }
    
private:
    void printStep(const std::string& name, duration_t time, int count, duration_t total) const {
        double avg = time.count() / static_cast<double>(count);
        double percent = 100.0 * time.count() / static_cast<double>(total.count());
        
        std::cout << "  " << std::left << std::setw(15) << name << ": " 
                  << std::right << std::setw(8) << time.count() << " | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << avg << " | "
                  << std::setw(6) << std::fixed << std::setprecision(2) << percent << "%\n";
    }
};

class AtomicBatchIncCore {
protected:
    Graph& G;
    Metric<int> MCD{0}, PCD{0};
    
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
        auto start = std::chrono::high_resolution_clock::now();
        
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
        
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<duration_t>(end - start);
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
    
    size_t estimateMemoryUsage() const {
        size_t total = 0;
        
        for (const auto& adjList : G.adj) {
            total += sizeof(adjList) + adjList.capacity() * sizeof(vertex_t);
        }
        total += G.core.capacity() * sizeof(int);
        
        total += MCD.values.capacity() * sizeof(int);
        total += PCD.values.capacity() * sizeof(int);
        
        return total;
    }
    
    BatchMetrics batchInsert(const std::vector<std::pair<vertex_t, vertex_t>>& edges, bool breakConnected) {
        BatchMetrics metrics;
        metrics.edgeCount = edges.size();
        
        auto totalStart = std::chrono::high_resolution_clock::now();
        
        if (breakConnected) {
            auto batches = decomposeEdges(edges);
            metrics.batchCount = batches.size();
            
            for (const auto& batch : batches) {
                auto batchMetrics = processBatchInsert(batch);
                
                metrics.graphUpdateTime += batchMetrics.graphUpdateTime;
                metrics.localUpdateTime += batchMetrics.localUpdateTime;
                metrics.buildCandTime += batchMetrics.buildCandTime;
                metrics.cascadeTime += batchMetrics.cascadeTime;
                metrics.rebuildTime += batchMetrics.rebuildTime;
                metrics.peakMemoryUsage = std::max(metrics.peakMemoryUsage, batchMetrics.peakMemoryUsage);
            }
        } else {
            metrics.batchCount = 1;
            auto batchMetrics = processBatchInsert(edges);
            
            metrics.graphUpdateTime = batchMetrics.graphUpdateTime;
            metrics.localUpdateTime = batchMetrics.localUpdateTime;
            metrics.buildCandTime = batchMetrics.buildCandTime;
            metrics.cascadeTime = batchMetrics.cascadeTime;
            metrics.rebuildTime = batchMetrics.rebuildTime;
            metrics.peakMemoryUsage = batchMetrics.peakMemoryUsage;
        }
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        metrics.totalTime = std::chrono::duration_cast<duration_t>(totalEnd - totalStart);
        
        return metrics;
    }
    
    BatchMetrics batchRemove(const std::vector<std::pair<vertex_t, vertex_t>>& edges, bool breakConnected) {
        BatchMetrics metrics;
        metrics.edgeCount = edges.size();
        
        auto totalStart = std::chrono::high_resolution_clock::now();
        
        if (breakConnected) {
            auto batches = decomposeEdges(edges);
            metrics.batchCount = batches.size();
            
            for (const auto& batch : batches) {
                auto batchMetrics = processBatchRemove(batch);
                
                metrics.graphUpdateTime += batchMetrics.graphUpdateTime;
                metrics.localUpdateTime += batchMetrics.localUpdateTime;
                metrics.buildCandTime += batchMetrics.buildCandTime;
                metrics.cascadeTime += batchMetrics.cascadeTime;
                metrics.rebuildTime += batchMetrics.rebuildTime;
                metrics.peakMemoryUsage = std::max(metrics.peakMemoryUsage, batchMetrics.peakMemoryUsage);
            }
        } else {
            metrics.batchCount = 1;
            auto batchMetrics = processBatchRemove(edges);
            
            metrics.graphUpdateTime = batchMetrics.graphUpdateTime;
            metrics.localUpdateTime = batchMetrics.localUpdateTime;
            metrics.buildCandTime = batchMetrics.buildCandTime;
            metrics.cascadeTime = batchMetrics.cascadeTime;
            metrics.rebuildTime = batchMetrics.rebuildTime;
            metrics.peakMemoryUsage = batchMetrics.peakMemoryUsage;
        }
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        metrics.totalTime = std::chrono::duration_cast<duration_t>(totalEnd - totalStart);
        
        return metrics;
    }
    
    std::vector<std::vector<std::pair<vertex_t, vertex_t>>> decomposeEdges(
        const std::vector<std::pair<vertex_t, vertex_t>>& edges) {
        
        std::vector<std::vector<std::pair<vertex_t, vertex_t>>> subBatches;
        
        std::unordered_map<vertex_t, int> vertexToBatch;
        
        for (const auto& [u, v] : edges) {
            bool uFound = vertexToBatch.find(u) != vertexToBatch.end();
            bool vFound = vertexToBatch.find(v) != vertexToBatch.end();
            
            if (!uFound && !vFound) {
                int newBatchIdx = subBatches.size();
                subBatches.push_back({});
                subBatches.back().push_back({u, v});
                vertexToBatch[u] = newBatchIdx;
                vertexToBatch[v] = newBatchIdx;
            } 
            else if (uFound && !vFound) {
                int batchIdx = vertexToBatch[u];
                subBatches[batchIdx].push_back({u, v});
                vertexToBatch[v] = batchIdx;
            } 
            else if (!uFound && vFound) {
                int batchIdx = vertexToBatch[v];
                subBatches[batchIdx].push_back({u, v});
                vertexToBatch[u] = batchIdx;
            } 
            else if (vertexToBatch[u] == vertexToBatch[v]) {
                int batchIdx = vertexToBatch[u];
                subBatches[batchIdx].push_back({u, v});
            } 
            else {
                int newBatchIdx = subBatches.size();
                subBatches.push_back({});
                subBatches.back().push_back({u, v});
                
                int uBatchIdx = vertexToBatch[u];
                int vBatchIdx = vertexToBatch[v];
                
                for (auto& [vertex, batchIdx] : vertexToBatch) {
                    if (batchIdx == uBatchIdx || batchIdx == vBatchIdx) {
                        batchIdx = newBatchIdx;
                    }
                }
                
                subBatches[newBatchIdx].insert(
                    subBatches[newBatchIdx].end(),
                    subBatches[uBatchIdx].begin(),
                    subBatches[uBatchIdx].end()
                );
                
                if (uBatchIdx != vBatchIdx) {
                    subBatches[newBatchIdx].insert(
                        subBatches[newBatchIdx].end(),
                        subBatches[vBatchIdx].begin(),
                        subBatches[vBatchIdx].end()
                    );
                }
                
                subBatches[uBatchIdx].clear();
                if (uBatchIdx != vBatchIdx) {
                    subBatches[vBatchIdx].clear();
                }
            }
        }
        
        std::vector<std::vector<std::pair<vertex_t, vertex_t>>> result;
        for (auto& batch : subBatches) {
            if (!batch.empty()) {
                result.push_back(std::move(batch));
            }
        }
        
        return result;
    }
    
private:
    BatchMetrics processBatchInsert(const std::vector<std::pair<vertex_t, vertex_t>>& edges) {
        BatchMetrics metrics;
        metrics.edgeCount = edges.size();
        metrics.batchCount = 1;
        
        auto graphUpdateStart = std::chrono::high_resolution_clock::now();
        for (const auto& [u, v] : edges) {
            G.addEdge(u, v);
            
            size_t need = std::max(u, v) + 1;
            if (G.core.size() < need) {
                G.core.resize(need, 0);
            }
        }
        auto graphUpdateEnd = std::chrono::high_resolution_clock::now();
        metrics.graphUpdateTime = std::chrono::duration_cast<duration_t>(graphUpdateEnd - graphUpdateStart);
        
        auto buildCandStart = std::chrono::high_resolution_clock::now();
        std::unordered_set<int> candidateVertices;
        
        for (const auto& [u, v] : edges) {
            addLocal(u, v);
            addLocal(v, u);
            
            if (G.core[u] > G.core[v]) {
                if (PCD.get(v) > G.core[v]) {
                    candidateVertices.insert(v);
                }
            } else if (G.core[u] < G.core[v]) {
                if (PCD.get(u) > G.core[u]) {
                    candidateVertices.insert(u);
                }
            } else {
                if (PCD.get(u) > G.core[u]) {
                    candidateVertices.insert(u);
                }
                if (PCD.get(v) > G.core[v]) {
                    candidateVertices.insert(v);
                }
            }
        }
        auto buildCandEnd = std::chrono::high_resolution_clock::now();
        metrics.buildCandTime = std::chrono::duration_cast<duration_t>(buildCandEnd - buildCandStart);
        metrics.localUpdateTime = metrics.buildCandTime;
        
        auto cascadeStart = std::chrono::high_resolution_clock::now();
        
        std::unordered_set<int> upgradeVertices;
        
        for (int v : candidateVertices) {
            int k = G.core[v];
            
            if (PCD.get(v) <= k) {
                continue;
            }
            
            std::unordered_set<int> connectedCore;
            std::unordered_set<int> evicted;
            
            {
                std::queue<int> q;
                std::unordered_set<int> seen;
                q.push(v);
                seen.insert(v);
                
                while (!q.empty()) {
                    int x = q.front();
                    q.pop();
                    
                    if (G.core[x] == k) {
                        connectedCore.insert(x);
                        
                        for (int w : G.adj[x]) {
                            if (G.core[w] == k && !seen.count(w)) {
                                seen.insert(w);
                                q.push(w);
                            }
                        }
                    }
                }
            }
            
            std::unordered_map<int, int> cd;
            for (int x : connectedCore) {
                cd[x] = PCD.get(x);
            }
            
            bool stable = false;
            while (!stable) {
                stable = true;
                
                for (int x : connectedCore) {
                    if (evicted.count(x)) {
                        continue;
                    }
                    
                    if (cd[x] <= k) {
                        evicted.insert(x);
                        stable = false;
                        
                        for (int w : G.adj[x]) {
                            if (connectedCore.count(w) && !evicted.count(w) && G.core[w] == k && qual(w, x)) {
                                cd[w]--;
                            }
                        }
                    }
                }
            }
            
            for (int x : connectedCore) {
                if (!evicted.count(x)) {
                    upgradeVertices.insert(x);
                }
            }
        }
        
        for (int v : upgradeVertices) {
            ++G.core[v];
        }
        
        auto cascadeEnd = std::chrono::high_resolution_clock::now();
        metrics.cascadeTime = std::chrono::duration_cast<duration_t>(cascadeEnd - cascadeStart);
        
        auto rebuildStart = std::chrono::high_resolution_clock::now();
        rebuild();
        auto rebuildEnd = std::chrono::high_resolution_clock::now();
        metrics.rebuildTime = std::chrono::duration_cast<duration_t>(rebuildEnd - rebuildStart);
        
        metrics.peakMemoryUsage = estimateMemoryUsage();
        
        return metrics;
    }
    
    BatchMetrics processBatchRemove(const std::vector<std::pair<vertex_t, vertex_t>>& edges) {
        BatchMetrics metrics;
        metrics.edgeCount = edges.size();
        metrics.batchCount = 1;
        
        auto localUpdateStart = std::chrono::high_resolution_clock::now();
        
        for (const auto& [u, v] : edges) {
            if (u >= G.n() || v >= G.n()) {
                continue;
            }
            
            delLocal(u, v);
            delLocal(v, u);
        }
        
        auto localUpdateEnd = std::chrono::high_resolution_clock::now();
        metrics.localUpdateTime = std::chrono::duration_cast<duration_t>(localUpdateEnd - localUpdateStart);
        
        auto buildCandStart = std::chrono::high_resolution_clock::now();
        
        std::map<int, std::vector<int>> kGroups;
        
        for (const auto& [u, v] : edges) {
            if (u >= G.n() || v >= G.n()) {
                continue;
            }
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku > 0) kGroups[ku].push_back(u);
            if (kv > 0) kGroups[kv].push_back(v);
        }
        
        auto buildCandEnd = std::chrono::high_resolution_clock::now();
        metrics.buildCandTime = std::chrono::duration_cast<duration_t>(buildCandEnd - buildCandStart);
        
        auto graphUpdateStart = std::chrono::high_resolution_clock::now();
        for (const auto& [u, v] : edges) {
            G.removeEdge(u, v);
        }
        auto graphUpdateEnd = std::chrono::high_resolution_clock::now();
        metrics.graphUpdateTime = std::chrono::duration_cast<duration_t>(graphUpdateEnd - graphUpdateStart);
        
        auto cascadeStart = std::chrono::high_resolution_clock::now();
        
        std::vector<int> kValues;
        for (const auto& [k, _] : kGroups) {
            kValues.push_back(k);
        }
        std::sort(kValues.begin(), kValues.end());
        
        for (int k : kValues) {
            std::vector<int>& seeds = kGroups[k];
            
            std::vector<int> candidates;
            std::unordered_map<int, int> deg;
            std::unordered_set<int> visited;
            
            for (int seed : seeds) {
                if (visited.count(seed) || G.core[seed] != k) continue;
                
                std::queue<int> q;
                q.push(seed);
                visited.insert(seed);
                
                while (!q.empty()) {
                    int v = q.front();
                    q.pop();
                    
                    if (G.core[v] == k) {
                        candidates.push_back(v);
                        
                        int eff_deg = 0;
                        for (int w : G.adj[v]) {
                            if (G.core[w] >= k) eff_deg++;
                        }
                        deg[v] = eff_deg;
                        
                        for (int w : G.adj[v]) {
                            if (G.core[w] == k && !visited.count(w)) {
                                visited.insert(w);
                                q.push(w);
                            }
                        }
                    }
                }
            }
            
            std::queue<int> cascadeQ;
            for (int v : candidates) {
                if (deg[v] < k) {
                    cascadeQ.push(v);
                }
            }
            
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
        
        auto cascadeEnd = std::chrono::high_resolution_clock::now();
        metrics.cascadeTime = std::chrono::duration_cast<duration_t>(cascadeEnd - cascadeStart);
        
        auto rebuildStart = std::chrono::high_resolution_clock::now();
        rebuild();
        auto rebuildEnd = std::chrono::high_resolution_clock::now();
        metrics.rebuildTime = std::chrono::duration_cast<duration_t>(rebuildEnd - rebuildStart);
        
        metrics.peakMemoryUsage = estimateMemoryUsage();
        
        return metrics;
    }
};

struct BatchProcessingExp {
    const std::vector<EdgeRaw>& E;
    size_t T = 1000;
    size_t startBin = 400;
    size_t W = 100;
    int slideCount = 5;
    
    std::vector<std::pair<size_t, size_t>> bins;
    std::vector<std::pair<int, int>> current_edges;
    int maxN = 0;
    
    BatchProcessingExp(const std::vector<EdgeRaw>& edges) : E(edges) {}
    
    void setParams(size_t binCount, size_t start, size_t windowSize, int slides) {
        T = binCount;
        startBin = start;
        W = windowSize;
        slideCount = slides;
    }
    
    void divideBins() {
        std::cout << "将边按时间戳分配到" << T << "个bin中..." << std::endl;
        
        long long min_ts = E.front().ts;
        long long max_ts = E.back().ts;
        double bin_width = double(max_ts - min_ts + 1) / T;
        
        std::cout << "时间范围: " << min_ts << " 到 " << max_ts << std::endl;
        std::cout << "每个bin的宽度: " << bin_width << std::endl;
        
        bins.resize(T);
        
        size_t current_idx = 0;
        for (size_t bin_id = 0; bin_id < T; ++bin_id) {
            long long bin_end_ts = min_ts + (bin_id + 1) * bin_width;
            
            size_t bin_start = current_idx;
            
            while (current_idx < E.size() && E[current_idx].ts < bin_end_ts) {
                current_idx++;
            }
            
            bins[bin_id] = {bin_start, current_idx};
            
            size_t bin_edge_count = current_idx - bin_start;
            if (bin_edge_count > 0 && (bin_id % 100 == 0 || bin_id == startBin || bin_id == startBin + W - 1)) {
                std::cout << "Bin " << bin_id << ": " << bin_edge_count << " 条边" << std::endl;
            }
        }
    }
    
    void buildWindow(size_t startBinPos) {
        std::cout << "构建窗口（从Bin " << startBinPos << " 开始，包含 " << W << " 个bin）..." << std::endl;
        
        current_edges.clear();
        maxN = 0;
        
        if (startBinPos >= bins.size()) {
            std::cerr << "起始bin超出范围！" << std::endl;
            return;
        }
        
        size_t endBin = std::min(startBinPos + W, bins.size());
        
        for (size_t bin_id = startBinPos; bin_id < endBin; ++bin_id) {
            auto [start, end] = bins[bin_id];
            for (size_t i = start; i < end; ++i) {
                current_edges.emplace_back(E[i].u, E[i].v);
                maxN = std::max({maxN, E[i].u, E[i].v});
            }
        }
        
        std::cout << "窗口构建完成: " << current_edges.size() << " 条边，最大顶点ID: " << maxN << std::endl;
    }

    void slidingWindowExperiment() {
        divideBins();
        
        std::cout << "\n=== 开始滑动窗口实验 (从Bin " << startBin << " 开始，滑动 " << slideCount << " 次) ===" << std::endl;
        
        struct SlideResult {
            size_t binPos;
            size_t edgeCount;
            int standardBatchCount;
            int nonConnectedBatchCount;
            duration_t standardTime;
            duration_t nonConnectedTime;
            Graph::CoreDifference coreDiff;
        };
        
        std::vector<SlideResult> results;
        
        for (int slide = 0; slide < slideCount; ++slide) {
            size_t currentStartBin = startBin + slide;
            
            std::cout << "\n== 滑动 #" << slide + 1 << " (Bin " << currentStartBin << ") ==" << std::endl;
            
            buildWindow(currentStartBin);
            
            if (current_edges.empty()) {
                std::cout << "窗口中没有边，跳过此次滑动" << std::endl;
                continue;
            }
            
            Graph baseGraph(maxN + 1);
            for (auto [u, v] : current_edges) {
                baseGraph.addEdge(u, v);
            }
            
            Graph g1 = baseGraph.clone();
            AtomicBatchIncCore inc1(g1);
            
            Graph g2 = baseGraph.clone();
            AtomicBatchIncCore inc2(g2);
            
            size_t nextBin = currentStartBin + W;
            if (nextBin >= bins.size()) {
                std::cout << "没有下一个bin用于测试，跳过此次滑动" << std::endl;
                continue;
            }
            
            std::vector<std::pair<int, int>> nextBinEdges;
            auto [start, end] = bins[nextBin];
            for (size_t i = start; i < end; ++i) {
                nextBinEdges.emplace_back(E[i].u, E[i].v);
                maxN = std::max({maxN, E[i].u, E[i].v});
            }
            
            std::cout << "下个Bin " << nextBin << " 有 " << nextBinEdges.size() << " 条边" << std::endl;
            
            if (nextBinEdges.empty()) {
                std::cout << "下一个bin中没有边，跳过此次滑动" << std::endl;
                continue;
            }
            
            auto standardStart = std::chrono::high_resolution_clock::now();
            auto standardMetrics = inc1.batchInsert(nextBinEdges, false);
            auto standardEnd = std::chrono::high_resolution_clock::now();
            duration_t standardTime = std::chrono::duration_cast<duration_t>(standardEnd - standardStart);
            
            auto nonConnectedStart = std::chrono::high_resolution_clock::now();
            auto nonConnectedMetrics = inc2.batchInsert(nextBinEdges, true);
            auto nonConnectedEnd = std::chrono::high_resolution_clock::now();
            duration_t nonConnectedTime = std::chrono::duration_cast<duration_t>(nonConnectedEnd - nonConnectedStart);
            
            auto coreDiff = g1.compareCores(g2);
            
            std::cout << "标准批处理: " << standardMetrics.edgeCount << " 条边, " 
                      << standardMetrics.batchCount << " 个批次, 耗时 " 
                      << standardTime.count() << " 微秒" << std::endl;
            
            std::cout << "非连通批处理: " << nonConnectedMetrics.edgeCount << " 条边, " 
                      << nonConnectedMetrics.batchCount << " 个批次, 耗时 " 
                      << nonConnectedTime.count() << " 微秒" << std::endl;
            
            std::cout << "核数差异: ";
            coreDiff.print();
            
            results.push_back({
                currentStartBin,
                nextBinEdges.size(),
                standardMetrics.batchCount,
                nonConnectedMetrics.batchCount,
                standardTime,
                nonConnectedTime,
                coreDiff
            });
            
            bool valid1 = g1.validateCore();
            bool valid2 = g2.validateCore();
            
            std::cout << "标准批处理核数验证: " << (valid1 ? "正确" : "有误") << std::endl;
            std::cout << "非连通批处理核数验证: " << (valid2 ? "正确" : "有误") << std::endl;
        }
        
        std::cout << "\n=== 滑动窗口实验结果摘要 ===" << std::endl;
        std::cout << "| Bin位置 | 边数 | 标准批次 | 非连通批次 | 标准时间(微秒) | 非连通时间(微秒) | 不匹配顶点 | 平均差异 |" << std::endl;
        std::cout << "|---------|------|----------|------------|----------------|------------------|------------|----------|" << std::endl;
        
        for (const auto& result : results) {
            std::cout << "| " << std::setw(7) << result.binPos 
                      << " | " << std::setw(4) << result.edgeCount 
                      << " | " << std::setw(8) << result.standardBatchCount 
                      << " | " << std::setw(10) << result.nonConnectedBatchCount 
                      << " | " << std::setw(14) << result.standardTime.count() 
                      << " | " << std::setw(16) << result.nonConnectedTime.count() 
                      << " | " << std::setw(10) << result.coreDiff.mismatchCount 
                      << " | " << std::setw(8) << std::fixed << std::setprecision(2) << result.coreDiff.avgDifference 
                      << " |" << std::endl;
        }
    }
};

std::vector<EdgeRaw> readFile(const std::string& p) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::ifstream in(p);
    if (!in) {
        std::cerr << "无法打开文件: " << p << std::endl;
        exit(1);
    }
    
    std::vector<EdgeRaw> e;
    EdgeRaw t;
    size_t line_count = 0;
    
    while (in >> t.u >> t.v >> t.ts) {
        ++line_count;
        if (t.u != t.v) {
            e.push_back(t);
        }
        
        if (line_count % 1000000 == 0) {
            std::cout << "已读取 " << line_count << " 行，有效边 " << e.size() << std::endl;
        }
    }
    
    std::sort(e.begin(), e.end(), [](auto& a, auto& b) {
        return a.ts < b.ts;
    });
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "文件读取完成: 总计 " << line_count << " 行，有效边 " << e.size() 
              << "，耗时 " << duration << " 毫秒" << std::endl;
    
    return e;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <边文件路径> [分区数] [起始bin] [窗口大小] [滑动次数]" << std::endl;
        return 1;
    }
    
    auto edges = readFile(argv[1]);
    std::cout << "从文件加载了 " << edges.size() << " 条边" << std::endl;
    
    BatchProcessingExp experiment(edges);
    
    if (argc >= 3) {
        experiment.T = std::stoi(argv[2]);
        std::cout << "设置分区数: " << experiment.T << std::endl;
    }
    
    if (argc >= 4) {
        experiment.startBin = std::stoi(argv[3]);
        std::cout << "设置起始bin: " << experiment.startBin << std::endl;
    }
    
    if (argc >= 5) {
        experiment.W = std::stoi(argv[4]);
        std::cout << "设置窗口大小: " << experiment.W << " 个bin" << std::endl;
    }
    
    if (argc >= 6) {
        experiment.slideCount = std::stoi(argv[5]);
        std::cout << "设置滑动次数: " << experiment.slideCount << std::endl;
    }
    
    experiment.slidingWindowExperiment();
    
    return 0;
}