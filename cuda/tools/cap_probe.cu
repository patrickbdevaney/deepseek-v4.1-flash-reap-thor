// cap_probe.cu — exhaustive per-feature capability probe for one CUDA target.
//
// Built after LOOP_LOG Findings 29 and 30, in which a capability was declared absent TWICE because
// the probe covered only one instruction family. The rule that came out of that: enumerate every
// family that could provide a capability before concluding any of them lacks it. This file is that
// enumeration, so the question is never answered from memory again.
//
// Each feature is one -D flag; the driver script compiles each in isolation and reports
// COMPILES / BLOCKED plus the SASS mnemonic actually emitted (which is what proves it is real).
//
//   bash scripts/cap_probe.sh [target]      (default sm_110a)
#include <cuda_runtime.h>
#include <cstdint>

__global__ void k(unsigned* o, const unsigned* i, float* f) {
    __shared__ alignas(128) unsigned smem[256];
    __shared__ alignas(8) unsigned long long bar;
    unsigned r = 0;

// ---------------- mma.sync family (Ampere/Ada/Hopper lineage) ----------------
#if   defined(C_MMA_F16_M16N8K16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+f"(f[0]):"r"(i[0]));
#elif defined(C_MMA_BF16_M16N8K16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+f"(f[0]):"r"(i[0]));
#elif defined(C_MMA_TF32_M16N8K8)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%0,%0,%0},{%1,%1},{%1},{%0,%0,%0,%0};":"+f"(f[0]):"r"(i[0]));
#elif defined(C_MMA_E4M3_M16N8K32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+f"(f[0]):"r"(i[0]));
#elif defined(C_MMA_E5M2_M16N8K32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+f"(f[0]):"r"(i[0]));
#elif defined(C_MMA_S8_M16N8K32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+r"(r):"r"(i[0]));
#elif defined(C_MMA_S4_M16N8K64)
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+r"(r):"r"(i[0]));
#elif defined(C_MMA_E2M1_M16N8K32)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0};":"+f"(f[0]):"r"(i[0]));
#elif defined(C_MMA_MXF4_BLOCKSCALE)
    asm volatile("mma.sync.aligned.kind::mxf4.block_scale.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 {%0,%0,%0,%0},{%1,%1,%1,%1},{%1,%1},{%0,%0,%0,%0},{%1},{0,0},{%1},{0,0};":"+f"(f[0]):"r"(i[0]));

// ---------------- wgmma family (Hopper lineage) ----------------
#elif defined(C_WGMMA_F16)
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%0,%0,%0}, %1, %1, 1, 1, 1, 0, 0;":"+f"(f[0]):"l"((unsigned long long)i));
#elif defined(C_WGMMA_FENCE)
    asm volatile("wgmma.fence.sync.aligned;");

// ---------------- tcgen05 family (Blackwell datacenter lineage) ----------------
#elif defined(C_T5_ALLOC)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"::"l"(smem),"r"(32));
#elif defined(C_T5_MMA_F16)
    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(C_T5_MMA_F8F6F4)
    asm volatile("tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(C_T5_MMA_MXF8F6F4)
    asm volatile("tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%0], [%0], 0;"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(C_T5_MMA_MXF4NVF4)
    asm volatile("tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%0], [%0], 0;"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(C_T5_MMA_CTA2)
    asm volatile("tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, 0;"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(C_T5_LD)
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];":"=r"(r):"r"(i[0]));
#elif defined(C_T5_ST)
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"::"r"(i[0]),"r"(i[1]));
#elif defined(C_T5_CP)
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;"::"r"(i[0]),"l"((unsigned long long)i));
#elif defined(C_T5_SHIFT)
    asm volatile("tcgen05.shift.cta_group::1.down [%0];"::"r"(i[0]));

// ---------------- TMA / bulk async copy ----------------
#elif defined(C_CPASYNC_CG)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"l"(smem),"l"(i));
#elif defined(C_CPASYNC_BULK)
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"l"(smem),"l"(i),"r"(256),"l"(&bar));
#elif defined(C_TMA_TILE_2D)
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];"::"l"(smem),"l"(i),"r"(0),"r"(0),"l"(&bar));
#elif defined(C_TMA_TILE_3D)
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%2,%2}],[%3];"::"l"(smem),"l"(i),"r"(0),"l"(&bar));
#elif defined(C_TMA_IM2COL)
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%0],[%1,{%2,%2,%2}],[%3],{%2};"::"l"(smem),"l"(i),"r"(0),"l"(&bar));
#elif defined(C_TMA_MULTICAST)
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%2,%2}],[%3],%4;"::"l"(smem),"l"(i),"r"(0),"l"(&bar),"h"((unsigned short)1));
#elif defined(C_TMA_STORE)
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0,{%1,%1}],[%2];"::"l"(i),"r"(0),"l"(smem));
#elif defined(C_TMA_REDUCE)
    asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0,{%1,%1}],[%2];"::"l"(i),"r"(0),"l"(smem));
#elif defined(C_TMA_PREFETCH)
    asm volatile("prefetch.tensormap [%0];"::"l"(i));

// ---------------- cluster / distributed shared memory ----------------
#elif defined(C_CLUSTER_BAR)
    asm volatile("barrier.cluster.arrive.aligned;");
#elif defined(C_DSMEM_MAPA)
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"((unsigned)(size_t)smem),"r"(0));
#elif defined(C_DSMEM_ST)
    asm volatile("st.async.shared::cluster.b32 [%0], %1;"::"r"((unsigned)(size_t)smem),"r"(i[0]));
#elif defined(C_CLUSTERID)
    asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(r));

// ---------------- async barriers / warp specialisation ----------------
#elif defined(C_MBARRIER)
    asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"l"(&bar));
#elif defined(C_MBARRIER_TRYWAIT)
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],0; selp.u32 %0,1,0,p;}":"=r"(r):"l"(&bar));
#elif defined(C_SETMAXNREG)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
#elif defined(C_ELECT)
    asm volatile("{.reg .pred p; .reg .b32 m; elect.sync m|p, 0xffffffff; selp.u32 %0,1,0,p;}":"=r"(r));
#elif defined(C_GRIDDEPCTRL)
    asm volatile("griddepcontrol.wait;");

// ---------------- conversions ----------------
#elif defined(C_CVT_E2M1_TO_F16X2)
    { __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)i[0], __NV_E2M1); r = *(unsigned*)&h; }
#elif defined(C_CVT_F32_TO_E2M1X2)
    asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;":"=h"(*(unsigned short*)&r):"f"(f[0]),"f"(f[1]));
#elif defined(C_CVT_E8M0)
    asm volatile("cvt.rn.satfinite.ue8m0x2.f32 %0, %1, %2;":"=h"(*(unsigned short*)&r):"f"(f[0]),"f"(f[1]));
#elif defined(C_CVT_E3M2)
    asm volatile("cvt.rn.satfinite.e3m2x2.f32 %0, %1, %2;":"=h"(*(unsigned short*)&r):"f"(f[0]),"f"(f[1]));
#elif defined(C_CVT_E2M3)
    asm volatile("cvt.rn.satfinite.e2m3x2.f32 %0, %1, %2;":"=h"(*(unsigned short*)&r):"f"(f[0]),"f"(f[1]));
#elif defined(C_CVT_FP8X4)
    asm volatile("cvt.rn.satfinite.e4m3x4.f32 %0, %1, %1, %1, %1;":"=r"(r):"f"(f[0]));

// ---------------- ldmatrix / stmatrix ----------------
#elif defined(C_LDMATRIX_X4)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%0,%0,%0},[%1];":"+r"(r):"l"(smem));
#elif defined(C_LDMATRIX_M16N16_B8)
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%0},[%1];":"+r"(r):"l"(smem));
#elif defined(C_STMATRIX)
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0],{%1,%1,%1,%1};"::"l"(smem),"r"(i[0]));

// ---------------- cache control / prefetch ----------------
#elif defined(C_L2_EVICT_POLICY)
    { unsigned long long p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(p)); o[0]=(unsigned)p; }
#elif defined(C_LD_L1_NOALLOC)
    asm volatile("ld.global.nc.L1::no_allocate.b32 %0,[%1];":"=r"(r):"l"(i));
#elif defined(C_PREFETCH_L2)
    asm volatile("prefetch.global.L2 [%0];"::"l"(i));
#elif defined(C_DISCARD)
    asm volatile("discard.global.L2 [%0], 128;"::"l"(i));
#elif defined(C_CPASYNC_L2HINT)
    { unsigned long long p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(p));
      asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"l"(smem),"l"(i),"l"(p)); }

// ---------------- misc arithmetic ----------------
#elif defined(C_DP4A)
    asm volatile("dp4a.s32.s32 %0,%1,%1,%0;":"+r"(r):"r"(i[0]));
#elif defined(C_REDUX)
    asm volatile("redux.sync.add.s32 %0,%1,0xffffffff;":"=r"(r):"r"(i[0]));
#elif defined(C_LOP3)
    asm volatile("lop3.b32 %0,%1,%1,%1,0xE8;":"=r"(r):"r"(i[0]));
#elif defined(C_PRMT)
    asm volatile("prmt.b32 %0,%1,%1,0x4140;":"=r"(r):"r"(i[0]));
#endif
    if (o) o[0] = r + (unsigned)f[0] + smem[0];
}
int main(){ return 0; }
