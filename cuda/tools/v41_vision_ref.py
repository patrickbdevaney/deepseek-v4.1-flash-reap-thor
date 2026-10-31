# V1 oracle: the checkpoint's own inference/vision.py ViT + Aligner (CPU) on one image -> files for tests/gate_vision.cu.
#   .venv/bin/python cuda/tools/v41_vision_ref.py IMAGE OUTDIR [max_n_token]
# OUTDIR/patches.bf16 (raw bf16 [P, 588], image_processor.load_image), meta.json {n_h, n_w, n_llm_h, n_llm_w},
# ref_f32.bin (fp32 model, the oracle) and ref_bf16.bin (the model's own bf16 path: the noise floor the CUDA
# tower must not exceed), both fp32 [n_llm_h * n_llm_w, 5120].
import json, os, sys, types
import numpy as np, torch
R = os.environ.get('V41_SRC', '/home/patrickd/models/DeepSeek-V4.1-Flash-1017')
sys.path.insert(0, R + '/inference')
import vision, image_processor as ip
from safetensors import safe_open
img, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(R + '/inference/config.json'))
if len(sys.argv) > 3: cfg['vision_max_n_token'] = int(sys.argv[3])
a = types.SimpleNamespace(**cfg)
torch.set_num_threads(8)
vit, al = vision.ViT(a), vision.Aligner(a)
wm = json.load(open(R + '/model.safetensors.index.json'))['weight_map']
sd = {}
for f in sorted(set(v for k, v in wm.items() if k.startswith(('vision', 'aligner')))):
    with safe_open(f'{R}/{f}', 'pt') as h:
        for k in h.keys():
            if k.startswith(('vision', 'aligner')): sd[k] = h.get_tensor(k)
vit.load_state_dict({k[7:]: v.float() for k, v in sd.items() if k.startswith('vision.')})
al.load_state_dict({k[8:]: v.float() for k, v in sd.items() if k.startswith('aligner.')})
patches, nh, nw, lh, lw = ip.load_image({'url': img}, a)
patches.contiguous().view(torch.int16).numpy().tofile(out + '/patches.bf16')
json.dump({'n_h': nh, 'n_w': nw, 'n_llm_h': lh, 'n_llm_w': lw}, open(out + '/meta.json', 'w'))
with torch.no_grad():
    e = al(vit(patches.float(), nh, nw), nh, nw)
e.numpy().astype(np.float32).tofile(out + '/ref_f32.bin')
vit16, al16 = vit.to(torch.bfloat16), al.to(torch.bfloat16)
with torch.no_grad():
    e16 = al16(vit16(patches, nh, nw), nh, nw).float()
e16.numpy().astype(np.float32).tofile(out + '/ref_bf16.bin')
c = torch.nn.functional.cosine_similarity(e16, e, dim=1)
print(f'patches {nh}x{nw} -> llm {lh}x{lw}; bf16 path vs fp32: cos min {c.min():.6f} mean {c.mean():.6f}')
