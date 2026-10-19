#include <cstdio>
__global__ void k(float* c, const unsigned* a, const unsigned* b){
#if defined(P_MXF4_M16N8K64)
  asm volatile("mma.sync.aligned.kind::mxf4.block_scale.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%10},{0,0};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(a[0]));
#elif defined(P_E2M1_M16N8K32)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#elif defined(P_E4M3_M16N8K32)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#elif defined(P_E2M1_M16N8K64)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#elif defined(P_CPASYNC_BULK)
  __shared__ unsigned smem[32];
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%0];\n"
    ::"l"(smem),"l"(a),"r"(128));
#elif defined(P_TCGEN05)
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::);
#elif defined(P_CVT_FP4X2)
  asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;\n" :"=r"(((unsigned*)c)[0]):"f"(c[1]),"f"(c[2]));
#elif defined(P_CVT_F16X2_E2M1X2)
  asm volatile("cvt.rn.f16x2.e2m1x2 %0, %1;\n" :"=r"(((unsigned*)c)[0]):"h"(*(const unsigned short*)a));
#endif
}
int main(){ printf("ok\n"); return 0; }
