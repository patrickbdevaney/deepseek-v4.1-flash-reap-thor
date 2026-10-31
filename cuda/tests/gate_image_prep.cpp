// gate_image_prep.cpp — V2: include/v41_image.h's preprocessing against image_processor.load_image's patches
// (tools/v41_vision_ref.py OUTDIR/patches.bf16 + meta.json) for the same file. Lossless input (PNG) -> must be
// bit-identical; JPEG may differ by stb vs libjpeg decoding only.
//   g++ -O2 -std=c++17 -I include tests/gate_image_prep.cpp -o build/gate_image_prep && build/gate_image_prep IMG REFDIR
#include "v41_image.h"
#include "third_party/json.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: gate_image_prep IMAGE REFDIR\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary); std::stringstream ss; ss << f.rdbuf();
    const auto P = v41img::prepare(v41img::decode(ss.str()));
    std::stringstream ms; ms << std::ifstream(std::string(argv[2]) + "/meta.json").rdbuf();
    const auto m = nlohmann::json::parse(ms.str());
    std::ifstream rp(std::string(argv[2]) + "/patches.bf16", std::ios::binary);
    std::vector<uint16_t> ref(P.patches.size());
    rp.read((char*)ref.data(), ref.size() * 2);
    const bool grid = m["n_h"] == P.n_vit_h && m["n_w"] == P.n_vit_w && m["n_llm_h"] == P.n_llm_h && m["n_llm_w"] == P.n_llm_w;
    long diff = 0, big = 0;
    for (size_t i = 0; i < ref.size(); ++i) if (ref[i] != P.patches[i]) { ++diff; if (std::abs((int)ref[i] - (int)P.patches[i]) > 4) ++big; }
    printf("grid %dx%d (llm %dx%d) %s; patches %zu values, %ld differ (%ld by > 4 bf16 ulps)\n", P.n_vit_h, P.n_vit_w, P.n_llm_h,
           P.n_llm_w, grid ? "matches" : "MISMATCH", ref.size(), diff, big);
    const bool ok = grid && (rp.gcount() == (std::streamsize)(ref.size() * 2)) && diff == 0;
    printf("IMAGE PREP %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
