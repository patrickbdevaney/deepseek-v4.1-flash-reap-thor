// load_device.cu — register the 96 GiB checkpoint for ZERO-COPY GPU access (unified memory) and verify
// the GPU reads a real weight correctly. Foundation for forward.cu (no separate 96 GiB device alloc).
//   build: nvcc -O2 -arch=sm_110a -I include tools/load_device.cu -o build/load_device
//   run:   ./build/load_device /home/patrickd/models/DeepSeek-V4-Flash-180B
#include "safetensors.h"
#include "weight_store.h"
#include "deepseek_v4.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

static std::string key_map(const std::string& in) {   // this REAP ckpt already uses attn/ffn names
    std::string s = in;
    if (s.rfind("model.", 0) == 0) s = s.substr(6);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <ckpt_dir>\n", argv[0]); return 2; }
    // --- probe device capabilities (Jetson is integrated: host mem IS device mem) ---
    int integrated=0, hostReg=0, canUseHostPtr=0, mapHost=0;
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0);
    cudaDeviceGetAttribute(&hostReg, cudaDevAttrHostRegisterSupported, 0);
    cudaDeviceGetAttribute(&canUseHostPtr, cudaDevAttrCanUseHostPointerForRegisteredMem, 0);
    cudaDeviceGetAttribute(&mapHost, cudaDevAttrCanMapHostMemory, 0);
    printf("integrated=%d hostRegisterSupported=%d canUseHostPtrForRegMem=%d canMapHostMem=%d\n",
           integrated, hostReg, canUseHostPtr, mapHost);

    // --- full-model load into GPU-accessible memory (single-copy pread; see weight_store.h) ---
    st::WeightStore W(argv[1], key_map);
    printf("loaded %.3f GiB, %zu tensors -> device pointers\n", W.loadedGiB(), W.count());

    // verify the GPU reads a real weight via its device pointer, matching the file bytes
    const char* probe = "layers.1.attn.wq_a.weight";
    if (!W.has(probe)) { fprintf(stderr, "probe tensor missing: %s\n", probe); return 1; }
    const st::DevTensor& dt = W.get(probe);
    unsigned char dbuf[32]; if (cudaMemcpy(dbuf, dt.dev, 32, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr,"D2H failed\n"); return 1; }
    // ground truth: re-read the same tensor's first 32 bytes straight from its shard mmap
    bool match = true;   // if the probe shard differs, this compares against W's own resolved pointer consistency
    { st::ShardedSafeTensors S2(argv[1], key_map); const st::Tensor& t = S2.get(probe);
      match = (memcmp(dbuf, t.data, 32) == 0); }
    printf("GPU read of %s[:32]: %s\n", probe, match ? "MATCH file bytes" : "MISMATCH");
    printf("\nFull-model weight->device load (single-copy pread, %.1f GiB): %s\n", W.loadedGiB(), match ? "PASS" : "FAIL");
    return match ? 0 : 1;
}
