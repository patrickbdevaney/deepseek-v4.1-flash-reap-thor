__global__ void k(unsigned* o, const unsigned* i){
#if defined(K_MXF4)
  asm volatile("tcgen05.mma.cta_group::1.kind::mxf4 [%0], %1, %2, %3, 0;\n"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(K_MXF4NVF4)
  asm volatile("tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%0], [%0], 0;\n"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(K_MXF8F6F4)
  asm volatile("tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%0], [%0], 0;\n"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(K_F8F6F4)
  asm volatile("tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 0;\n"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(K_F16)
  asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"::"r"(i[0]),"l"(i),"l"(i),"r"(i[1]));
#elif defined(K_LDMATRIX_B4)
  unsigned r; asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64.shared.b8 {%0}, [%1];\n":"=r"(r):"l"(i)); o[0]=r;
#elif defined(K_LDMATRIX_M16N16_B8)
  unsigned r; asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0}, [%1];\n":"=r"(r):"l"(i)); o[0]=r;
#endif
}
