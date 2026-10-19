# cutlass_moe.cu — CUTLASS NVFP4 TC GEMM wrapper (STEP 1 DONE, validated 2026-07-01)
Callable API (extern "C"):
  cutlass_nvfp4_gemm(D_bf16, A_fp4, B_fp4, SFA_swz, SFB_swz, alpha, M,N,K, workspace, stream) -> rc(0=ok)
  cutlass_sfa_bytes/cutlass_sfb_bytes/cutlass_workspace_bytes(M,N,K)
VALIDATED: all-1.0 input (E2M1 code2=1.0 byte 0x22, ue4m3 1.0 = byte 0x38) -> D = K = 2816.0 EXACT.
  scale 0x40(=2.0) -> D=K*4. Numerically correct on Thor sm_110a / CUDA 13.0. Tile 128x128x256 1SM.
Build test: nvcc -std=c++17 --expt-relaxed-constexpr -DCUTLASS_MOE_TEST -I ~/cutlass/include
  -I ~/cutlass/tools/util/include -gencode arch=compute_110a,code=sm_110a kernels/cutlass_moe.cu -o test
Link into server: compile WITHOUT -DCUTLASS_MOE_TEST -> object; link with forward.o.
NEXT (step 2): convert our E2M1 + LINEAR e4m3 group-16 scales -> CUTLASS swizzled SFA/SFB layout
  (Sm1xxBlkScaledConfig::tile_atom_to_shape_SF*). Validated so far only with UNIFORM scales (swizzle-agnostic).

## STEP 2 DONE (2026-07-01): scale swizzle validated
- cutlass_swizzle_sfa/sfb(out, in_linear, M,N,K, stream): converts our LINEAR e4m3 [rows][K/16] scales to
  CUTLASS blocked layout via the cute layout (3-arg call lay(row, k_elem, 0) -> swizzled offset).
- SFB blocked layout = (((32,4),Nblk),((16,4),Kblk),(1,L)) : 128x4 blocks, 512-elem stride (from cute print).
- END-TO-END VALIDATED: random FP4 codes + random scales -> swizzle -> wrapper vs fp32 reference:
  M16xN2816xK704 (down), M16xN704xK2816 (gateup), M128 square -> ALL maxrel 0.0039 (bf16 rounding), 0 bad.
- NEXT (step 3): GROUPED GEMM (kGrouped, ex.92 API) for the 8-expert verify MoE in ONE launch (per-expert
  single GEMM = 79 launches x 41us overhead > half2; MUST use grouped). Then quantize activation to FP4 +
  convert expert weights to CUTLASS layout at load. Then wire into moe() for seq<=16.
