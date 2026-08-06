#include "../cactus_engine.h"
#include "test_utils.h"
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <random>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

const char* g_index_path = std::getenv("CACTUS_INDEX_PATH");
constexpr size_t DIM = 1024;

std::vector<float> random_embedding(size_t dim = DIM) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> emb(dim);
    for (auto& v : emb) v = dist(gen);
    return emb;
}

class IndexFixture {
public:
    explicit IndexFixture(const std::string& name, size_t dim = DIM, bool fresh = true)
        : dir_(std::string(g_index_path) + "/" + name), dim_(dim), idx_(nullptr), cleanup_(fresh) {
        if (fresh) {
            cleanup();
            mkdir(dir_.c_str(), 0755);
        }
    }

    ~IndexFixture() {
        if (idx_) cactus_index_destroy(idx_);
        if (cleanup_) cleanup();
    }

    void keep_files() { cleanup_ = false; }

    cactus_index_t init() { return idx_ = cactus_index_init(dir_.c_str(), dim_); }

    cactus_index_t reopen() {
        if (idx_) { cactus_index_destroy(idx_); idx_ = nullptr; }
        return init();
    }

    int add(int id, const char* content, const std::vector<float>& emb) {
        const float* p = emb.data();
        const char* meta = "meta";
        return cactus_index_add(idx_, &id, &content, &meta, &p, 1, dim_);
    }

    int add(int id, const char* content) { return add(id, content, random_embedding(dim_)); }

    int add_batch(int start, int count) {
        std::vector<int> ids(count);
        std::vector<std::string> docs(count), metas(count);
        std::vector<const char*> doc_ptrs(count), meta_ptrs(count);
        std::vector<std::vector<float>> embs(count);
        std::vector<const float*> emb_ptrs(count);

        for (int i = 0; i < count; ++i) {
            ids[i] = start + i;
            docs[i] = "doc" + std::to_string(ids[i]);
            metas[i] = "meta";
            doc_ptrs[i] = docs[i].c_str();
            meta_ptrs[i] = metas[i].c_str();
            embs[i] = random_embedding(dim_);
            emb_ptrs[i] = embs[i].data();
        }
        return cactus_index_add(idx_, ids.data(), doc_ptrs.data(), meta_ptrs.data(), emb_ptrs.data(), count, dim_);
    }

    std::pair<int, std::string> get(int id) {
        char* buf = (char*)malloc(65536);
        size_t size = 65536;
        int r = cactus_index_get(idx_, &id, 1, &buf, &size, nullptr, nullptr, nullptr, nullptr);
        std::string content = (r == 0) ? std::string(buf) : "";
        free(buf);
        return {r, content};
    }

    int del(int id) { return cactus_index_delete(idx_, &id, 1); }
    int del(const std::vector<int>& ids) { return cactus_index_delete(idx_, ids.data(), ids.size()); }
    int compact() { return cactus_index_compact(idx_); }

    std::vector<int> query(const std::vector<float>& emb, int k = 10) {
        const float* p = emb.data();
        int* ids = (int*)malloc(k * sizeof(int));
        float* scores = (float*)malloc(k * sizeof(float));
        size_t id_sz = k, sc_sz = k;
        std::string opts = "{\"top_k\":" + std::to_string(k) + "}";
        cactus_index_query(idx_, &p, 1, dim_, opts.c_str(), &ids, &id_sz, &scores, &sc_sz);
        std::vector<int> result(ids, ids + id_sz);
        free(ids); free(scores);
        return result;
    }

    const std::string& path() const { return dir_; }
    cactus_index_t get_idx() const { return idx_; }

private:
    void cleanup() {
        unlink((dir_ + "/index.bin").c_str());
        unlink((dir_ + "/data.bin").c_str());
        unlink((dir_ + "/index.bin.backup").c_str());
        unlink((dir_ + "/data.bin.backup").c_str());
        rmdir(dir_.c_str());
    }
    std::string dir_;
    size_t dim_;
    cactus_index_t idx_;
    bool cleanup_;
};

bool test_crud() {
    IndexFixture f("test_crud");
    if (!f.init()) return false;

    if (f.add(1, "hello") != 0) return false;
    if (f.add(2, "world") != 0) return false;

    auto [r1, c1] = f.get(1);
    auto [r2, c2] = f.get(2);
    if (r1 != 0 || c1 != "hello") return false;
    if (r2 != 0 || c2 != "world") return false;

    if (f.del(1) != 0) return false;
    if (f.get(1).first == 0) return false;  
    if (f.get(2).first != 0) return false; 

    return true;
}

bool test_batch() {
    IndexFixture f("test_batch");
    if (!f.init()) return false;

    if (f.add_batch(0, 100) != 0) return false;

    if (f.get(0).second != "doc0") return false;
    if (f.get(50).second != "doc50") return false;
    if (f.get(99).second != "doc99") return false;

    if (f.del({10, 20, 30, 40, 50}) != 0) return false;
    if (f.get(50).first == 0) return false;  
    if (f.get(51).first != 0) return false;  

    return true;
}

bool test_query() {
    IndexFixture f("test_query");
    if (!f.init()) return false;

    auto target_emb = random_embedding();
    f.add(1, "target", target_emb);
    for (int i = 2; i <= 10; ++i) f.add(i, "other");

    auto results = f.query(target_emb, 1);
    if (results.empty() || results[0] != 1) return false;

    f.del(1);
    results = f.query(target_emb, 10);
    for (int id : results) {
        if (id == 1) return false;  
    }

    return true;
}

bool test_compact() {
    IndexFixture f("test_compact");
    if (!f.init()) return false;

    f.add_batch(0, 20);

    struct stat st_before;
    stat((f.path() + "/index.bin").c_str(), &st_before);

    f.del({0, 2, 4, 6, 8, 10, 12, 14, 16, 18});  // Delete even IDs
    f.compact();

    struct stat st_after;
    stat((f.path() + "/index.bin").c_str(), &st_after);

    if (st_after.st_size >= st_before.st_size) return false;

    if (f.get(1).second != "doc1") return false;
    if (f.get(19).second != "doc19") return false;

    if (f.get(0).first == 0) return false;

    return true;
}

bool test_persistence() {
    IndexFixture f("test_persist");
    if (!f.init()) return false;

    f.add_batch(0, 10);
    f.del(5);
    f.compact();

    if (!f.reopen()) return false;

    if (f.get(0).second != "doc0") return false;
    if (f.get(5).first == 0) return false; 
    if (f.get(9).second != "doc9") return false;

    return true;
}

bool test_errors() {
    IndexFixture f("test_errors");
    if (!f.init()) return false;

    f.add(1, "first");
    if (f.add(1, "second") == 0) return false;

    if (f.get(999).first == 0) return false;

    if (f.del(999) == 0) return false;

    f.del(1);
    if (f.del(1) == 0) return false;

    std::vector<float> zero_emb(DIM, 0.0f);
    if (f.add(2, "zero", zero_emb) == 0) return false;

    return true;
}

bool test_unicode() {
    IndexFixture f("test_unicode");
    if (!f.init()) return false;

    f.add(1, "Hello 世界 🌍");
    return f.get(1).second == "Hello 世界 🌍";
}

bool test_constructor() {
    {
        IndexFixture f("test_ctor_valid");
        if (!f.init()) return false;
    }

    {
        IndexFixture f("test_ctor_dim");
        if (!f.init()) return false;
        f.add(1, "test");
        f.keep_files();
    }
    {
        IndexFixture f("test_ctor_dim", 256, false);  
        bool failed = (f.init() == nullptr);  
        std::string dir = f.path();
        unlink((dir + "/index.bin").c_str());
        unlink((dir + "/data.bin").c_str());
        rmdir(dir.c_str());
        if (!failed) return false;
    }

    return true;
}

int main() {
    TestUtils::TestRunner runner("Index Tests");

    runner.run_test("crud", test_crud());
    runner.run_test("batch", test_batch());
    runner.run_test("query", test_query());
    runner.run_test("compact", test_compact());
    runner.run_test("persistence", test_persistence());
    runner.run_test("errors", test_errors());
    runner.run_test("unicode", test_unicode());
    runner.run_test("constructor", test_constructor());

    runner.print_summary();

    return runner.all_passed() ? 0 : 1;
}
