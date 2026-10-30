// v41-vision-dump: encode one image through the mmproj alone and write the LLM-side token block.
//
// no text model is loaded (mtmd accepts a null one); the block is the full
//   [START] ([IMAGE]*n_llm_w [NEWLINE])*n_llm_h [END]
// sequence the decoder would receive, in order, as f32 [n_tok, dim].
//
// usage: v41-vision-dump MMPROJ.gguf IMAGE OUT.f32 DIM [n_threads] [gpu=0]
// writes OUT.f32 and prints "n_tok=<n> dim=<d>"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s MMPROJ.gguf IMAGE OUT.f32 DIM [n_threads] [gpu=0]\n", argv[0]);
        return 2;
    }
    const int dim       = atoi(argv[4]);
    const int n_threads = argc > 5 ? atoi(argv[5]) : 8;
    const bool use_gpu  = argc > 6 && atoi(argv[6]) != 0;

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu       = use_gpu;
    mp.n_threads     = n_threads;
    mp.print_timings = true;
    mp.warmup        = false;
    mtmd_context * ctx = mtmd_init_from_file(argv[1], nullptr, mp);
    if (!ctx) { fprintf(stderr, "mmproj load failed\n"); return 1; }

    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(ctx, argv[2], false, mtmd_helper_init_opt_default()).bitmap;
    if (!bmp) { fprintf(stderr, "image load failed\n"); return 1; }

    mtmd_input_part part = {};
    part.bitmap = bmp;
    const mtmd_input_part * parts[] = { &part };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize_from_parts(ctx, chunks, parts, 1, false) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img = c; }
    }
    if (!img) { fprintf(stderr, "no image chunk\n"); return 1; }
    if (mtmd_encode_chunk(ctx, img) != 0) { fprintf(stderr, "encode failed\n"); return 1; }

    const size_t n_tok = mtmd_input_chunk_get_n_tokens(img);
    const float * embd = mtmd_get_output_embd(ctx);
    FILE * f = fopen(argv[3], "wb");
    if (!f || fwrite(embd, sizeof(float), n_tok * dim, f) != n_tok * dim) { fprintf(stderr, "write failed\n"); return 1; }
    fclose(f);
    printf("n_tok=%zu dim=%d\n", n_tok, dim);

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    mtmd_free(ctx);
    return 0;
}
