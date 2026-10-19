# Engram row file (`.v41rows`), format v1

DeepSeek-V4.1-Flash has two Engram tables: layer 1 has 384,006,168 rows and layer 14 has 384,016,682 rows. Each row holds 256 FP8 E4M3 values and 8 E8M0 scales, one per 32 values. Each table is about 101 GB, so neither the GGUF nor host memory holds it.

A token reads 24 rows per Engram layer. The row ids are a pure function of the token ids (the n-gram hash). That means they can be read with `O_DIRECT` from NVMe while the GPU runs. Measured on this Thor: 48 rows per token in 1.13–1.30 ms p50 with 16 threads.

This file is the single on-disk format. The llama.cpp fork (`src/models/deepseek41-engram-rowstore.h`) reads it, and so should the CUDA server (`cuda/`). `ops/s09_engram_rowfile.py` builds and verifies it from base shards 47/48. The files are never pruned or quantized: Engram is excluded from REAP, and its rows are bit-identical to the teacher's.

## Layout

There is one file per Engram layer, named `engram_L<layer>.v41rows` (for example `engram_L1.v41rows`). All integers are little-endian.

| offset | size | field | value |
|---|---|---|---|
| 0 | 8 | magic | `V41ENGR1` |
| 8 | 4 | u32 version | 1 |
| 12 | 4 | u32 layer_id | decoder layer (1 or 14) |
| 16 | 8 | u64 n_rows | table rows (the full table: 384,006,168 / 384,016,682) |
| 24 | 4 | u32 key_len | values per row, 256 |
| 28 | 4 | u32 n_scales | scales per row, 8 (one per 32 values) |
| 32 | 4 | u32 row_bytes | key_len + n_scales = 264 |
| 36 | 4 | u32 value_dtype | 1 = FP8 E4M3FN |
| 40 | 4 | u32 scale_dtype | 1 = E8M0 (value = 2^(e-127), 0xFF = NaN) |
| 44 | 8 | u64 data_off | 4096 |
| 52 | 32 | sha256 of the source safetensors shard holding this table | |
| 84 | 64 | ASCII source tensor name, NUL padded | `layers.<L>.engram.embed.weight` |
| 148 | 40 | ASCII teacher revision, NUL padded | `1bae0e10…` |
| 188 | 32 | sha256 of the data region (`data_off .. data_off + n_rows*row_bytes`) | written last |
| 220 | … | zero padding to 4096 | |

Row `r` is stored at `data_off + r * 264`:

```
uint8 values[256]   FP8 E4M3FN, row r of layers.<L>.engram.embed.weight   (bytes copied, not decoded)
uint8 scales[8]     E8M0,       row r of layers.<L>.engram.embed.scale
```

Rows are packed with no padding. A 264 B row therefore crosses at most one 512 B sector boundary, so one aligned read of 1 KiB always covers it.

## Dequantisation

The reference is `ParallelEngramEmbedding.forward`:

```
v[c] = bf16( float(fp8[c]) * 2^(scale[c/32] - 127) )
```

The value is computed in fp32 and then rounded to bf16, round-to-nearest-even. The llama.cpp reader returns those bf16 values widened to f32, so its `engram_embd` input is bit-identical to the reference's.

## Why a side file, and not the GGUF offset

The GGUF route would leave the table inside the GGUF and pread it from its data offset. That has three problems:

- It ties a 200 GB table to every quant level.
- GGUF data alignment is 32 B, so rows land at arbitrary sector phases.
- The PR's converter re-quantizes the table (Q8_0) instead of copying FP8 bits.

The side file is built once, is shared by every GGUF level and by the CUDA server, and is bit-exact.

## Reader contract

- Open with `O_RDONLY|O_DIRECT`. If the filesystem refuses `O_DIRECT`, fall back to buffered I/O; the result is correct but pollutes the page cache.
- Read the aligned window `[floor(off/A)*A, ceil((off+264)/A)*A)` with `A` = `LLAMA_ENGRAM_ALIGN` (default 4096, minimum 512).
- Readers must check magic, version, layer_id, `row_bytes == key_len + n_scales` and `data_off % 4096 == 0`. They must also check that `n_rows` covers the largest bucket (`max(offset + prime)` from the GGUF engram constants). Any mismatch is fatal.
- llama.cpp enables the row store with `LLAMA_ENGRAM_ROWFILE_DIR=<dir>`. The GGUF then needs no `engram_embd` tensors; s09 GGUFs ship without them.

## Storage

The two files total 202.8 GB (about 188.8 GiB). They must be on NVMe. The Seagate USB/NTFS drive is for convenience copies only and is never served from (see the project memory). The header's data sha256 lets a copy be verified without the source shards.
