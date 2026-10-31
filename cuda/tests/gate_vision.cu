// gate_vision.cu — V1: the CUDA vision tower (v41_vision.h) against the fp32 torch oracle (tools/v41_vision_ref.py).
//   build/gate_vision CKPT REFDIR [reps]
// Pass bar (as the llama.cpp golden, llamacpp/tests/v41_vision_golden.py): min cos and mean rel-L2 vs fp32 no worse
// than the model's own bf16 path vs fp32. Also times encode_image.
#include "v41_vision.h"
#include "third_party/json.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
template <class T> static std::vector<T> rd(const std::string& f) {
    std::ifstream s(f, std::ios::binary); s.seekg(0, std::ios::end); std::vector<T> v(s.tellg() / sizeof(T)); s.seekg(0);
    s.read((char*)v.data(), v.size() * sizeof(T)); return v;
}
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: gate_vision CKPT REFDIR [reps]\n"); return 2; }
    const std::string ref = argv[2];
    std::stringstream ss; ss << std::ifstream(ref + "/meta.json").rdbuf();
    const auto meta = nlohmann::json::parse(ss.str());
    const int nh = meta["n_h"], nw = meta["n_w"], L = v41v::llm_rows(nh, nw);
    const auto pt = rd<uint16_t>(ref + "/patches.bf16"); const auto r32 = rd<float>(ref + "/ref_f32.bin"), r16 = rd<float>(ref + "/ref_bf16.bin");
    if ((int)pt.size() != nh * nw * v41v::PATCH_IN || (int)r32.size() != L * v41v::LLM_DIM) { fprintf(stderr, "size mismatch\n"); return 2; }
    v41v::Vision V;
    v41v::load_vision(V, argv[1], std::max(nh * nw, 1024));
    printf("[vision] loaded %.2f GiB; image %dx%d patches -> %d rows\n", V.gib, nh, nw, L);
    uint16_t* dp; float* dout; cudaMalloc(&dp, pt.size() * 2); cudaMalloc(&dout, (size_t)L * v41v::LLM_DIM * 4);
    cudaMemcpy(dp, pt.data(), pt.size() * 2, cudaMemcpyHostToDevice);
    v41v::encode_image(V, dp, nh, nw, dout, 0);
    if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    const int reps = argc > 3 ? atoi(argv[3]) : 3;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) v41v::encode_image(V, dp, nh, nw, dout, 0);
    cudaDeviceSynchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    std::vector<float> o((size_t)L * v41v::LLM_DIM); cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost);
    auto stats = [&](const std::vector<float>& a, double& cmin, double& relm) {
        cmin = 1; relm = 0;
        for (int r = 0; r < L; ++r) {
            double d = 0, na = 0, nb = 0, e = 0;
            for (int c = 0; c < v41v::LLM_DIM; ++c) {
                const double x = a[(size_t)r * v41v::LLM_DIM + c], y = r32[(size_t)r * v41v::LLM_DIM + c];
                d += x * y; na += x * x; nb += y * y; e += (x - y) * (x - y);
            }
            cmin = std::min(cmin, d / std::sqrt(na * nb + 1e-30)); relm += std::sqrt(e / (nb + 1e-30)) / L;
        }
    };
    double c1, r1, c2, r2; stats(o, c1, r1); stats(r16, c2, r2);
    bool fin = true; for (float v : o) fin = fin && std::isfinite(v);
    printf("[vision] cuda vs fp32: cos min %.6f rel-L2 mean %.5f | bf16 ref vs fp32: cos min %.6f rel-L2 mean %.5f | %.1f ms/image\n",
           c1, r1, c2, r2, ms);
    const bool ok = fin && c1 >= c2 && r1 <= r2;
    printf("VISION GATE %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
