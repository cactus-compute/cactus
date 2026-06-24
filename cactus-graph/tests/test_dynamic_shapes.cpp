#include "test_utils.h"
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

using namespace TestUtils;

namespace {

bool test_dynamic_batch_matches_static() {
    const size_t K = 8, N = 8, MAXB = 4;
    std::vector<__fp16> wd(K * N), bd(N), xd(MAXB * K);
    fill_random_fp16(wd);
    fill_random_fp16(bd);
    fill_random_fp16(xd);

    CactusGraph dyn;
    size_t x = dyn.input({1, K}, Precision::FP16);
    size_t w = dyn.input({K, N}, Precision::FP16);
    size_t mm = dyn.matmul(x, w, false);
    size_t bias = dyn.input({1, N}, Precision::FP16);
    size_t out = dyn.add(mm, bias);

    for (size_t B : {size_t(1), size_t(4), size_t(2)}) {
        std::vector<__fp16> xb(xd.begin(), xd.begin() + B * K);

        dyn.set_runtime_input_shape(x, {B, K});
        dyn.set_input(x, xb.data(), Precision::FP16);
        dyn.set_input(w, wd.data(), Precision::FP16);
        dyn.set_input(bias, bd.data(), Precision::FP16);
        dyn.execute();

        if (dyn.get_output_buffer(out).shape != std::vector<size_t>{B, N}) return false;
        std::vector<float> dvals(B * N);
        const __fp16* dop = static_cast<__fp16*>(dyn.get_output(out));
        for (size_t i = 0; i < B * N; ++i) dvals[i] = static_cast<float>(dop[i]);

        CactusGraph st;
        size_t sx = st.input({B, K}, Precision::FP16);
        size_t sw = st.input({K, N}, Precision::FP16);
        size_t smm = st.matmul(sx, sw, false);
        size_t sbias = st.input({1, N}, Precision::FP16);
        size_t sout = st.add(smm, sbias);
        st.set_input(sx, xb.data(), Precision::FP16);
        st.set_input(sw, wd.data(), Precision::FP16);
        st.set_input(sbias, bd.data(), Precision::FP16);
        st.execute();

        if (st.get_output_buffer(sout).shape != std::vector<size_t>{B, N}) return false;
        const __fp16* sop = static_cast<__fp16*>(st.get_output(sout));
        for (size_t i = 0; i < B * N; ++i) {
            if (std::abs(dvals[i] - static_cast<float>(sop[i])) > 1e-3f) return false;
        }
        st.hard_reset();
    }
    dyn.hard_reset();
    return true;
}

bool test_buffer_resize_tracks_bucket() {
    BufferPool pool;
    BufferDesc d({4}, Precision::FP16);
    d.resize_from_pool(pool);
    if (d.get_data() == nullptr || d.byte_size != 8) return false;

    d.set_shape({4096});
    d.resize_from_pool(pool);
    if (d.byte_size != 8192 || d.get_data() == nullptr) return false;

    d.set_shape({4});
    d.resize_from_pool(pool);
    if (d.byte_size != 8 || d.get_data() == nullptr) return false;
    return true;
}

bool test_dynamic_mask_roundtrip() {
    CactusGraph g;
    size_t x = g.input({1, 8}, Precision::FP16);
    size_t w = g.input({8, 8}, Precision::FP16);
    size_t mm = g.matmul(x, w, false);
    size_t bias = g.input({1, 8}, Precision::FP16);
    g.add(mm, bias);
    g.set_runtime_input_shape(x, {1, 8});

    std::string path =
        (std::filesystem::temp_directory_path() / "cactus_dynamic_roundtrip.cactus").string();
    GraphFile::save_graph(g, path);
    GraphFile::SerializedGraph sg = GraphFile::load_graph(path);
    std::filesystem::remove(path);

    if (sg.header.version != 6) return false;
    size_t dynamic_inputs = 0;
    for (const auto& n : sg.nodes) {
        if (n.op_type == OpType::INPUT && !n.dynamic_mask.empty()) dynamic_inputs++;
    }
    return dynamic_inputs == 1;
}

}

int main() {
    TestUtils::TestRunner runner("Dynamic Shape Tests");
    runner.run_test("Dynamic batch matches static", test_dynamic_batch_matches_static());
    runner.run_test("Buffer resize tracks bucket", test_buffer_resize_tracks_bucket());
    runner.run_test("Dynamic mask roundtrip", test_dynamic_mask_roundtrip());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
