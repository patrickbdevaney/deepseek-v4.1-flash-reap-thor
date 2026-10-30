# V4.1 vision golden: llama.cpp mmproj block (v41-vision-dump) vs the torch f32 reference ViT+Aligner, CPU.
# usage: python v41_vision_golden.py IMAGE CPP_DUMP.f32   (dump: v41-vision-dump MMPROJ IMAGE OUT.f32 5120)
# pass bar: cpp-vs-f32 error no worse than the reference bf16 path vs f32 (min cos, mean rel-L2).
import json, sys, types, torch, numpy as np
R = '/home/patrickd/models/DeepSeek-V4.1-Flash-1017'
sys.path.insert(0, R + '/inference')
import vision, image_processor as ip
from safetensors import safe_open
img_path, cpp_path = sys.argv[1], sys.argv[2]
cfg = json.load(open(R + '/inference/config.json'))
a = types.SimpleNamespace(**cfg)
torch.set_num_threads(12)
vit, al = vision.ViT(a), vision.Aligner(a)
wm = json.load(open(R + '/model.safetensors.index.json'))['weight_map']
sd = {}
for f in sorted(set(v for k, v in wm.items() if k.startswith(('vision', 'aligner', 'image_')))):
    with safe_open(f'{R}/{f}', 'pt') as h:
        for k in h.keys():
            if k.startswith(('vision', 'aligner', 'image_')):
                sd[k] = h.get_tensor(k).float()
print('dtypes ok', len(sd))
vit.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith('vision.')})
al.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith('aligner.')})
patches, nvh, nvw, nlh, nlw = ip.load_image({'url': img_path}, a)
with torch.no_grad():
    e = al(vit(patches.float(), nvh, nvw), nvh, nvw)
t = ip.image_token_types(nlh, nlw)
blk = torch.zeros(len(t), a.dim)
blk[t == ip.IMAGE_START] = sd['image_start']; blk[t == ip.IMAGE_END] = sd['image_end']
blk[t == ip.IMAGE_NEW_LINE] = sd['image_newline']; blk[t == ip.IMAGE] = e
c = torch.from_numpy(np.fromfile(cpp_path, np.float32))
print(f'grid vit {nvh}x{nvw} llm {nlh}x{nlw} ref n_tok {len(t)} cpp n_tok {c.numel() // a.dim}')
c = c.view(-1, a.dim)
assert c.shape == blk.shape
cos = torch.nn.functional.cosine_similarity(c, blk, dim=1)
d = (c - blk).abs()
rel = d.norm(dim=1) / blk.norm(dim=1)
im = t == ip.IMAGE
print(f'IMAGE rows: cos min {cos[im].min():.6f} mean {cos[im].mean():.6f}  rel-L2 max {rel[im].max():.4f} mean {rel[im].mean():.4f}  maxabs {d[im].max():.4g} (ref absmax {blk[im].abs().max():.4g})')
print(f'sentinel rows: cos min {cos[~im].min():.6f} maxabs {d[~im].max():.4g}')
worst = cos.argmin().item(); print('worst row', worst, 'type', t[worst].item(), f'{cos[worst]:.6f}')
# noise floor: the reference's own bf16 path (weights bf16 as shipped) vs f32
vit16, al16 = vit.to(torch.bfloat16), al.to(torch.bfloat16)
with torch.no_grad():
    e16 = al16(vit16(patches, nvh, nvw), nvh, nvw).float()
c16 = torch.nn.functional.cosine_similarity(e16, e, dim=1); r16 = (e16 - e).norm(dim=1) / e.norm(dim=1)
ccp = cos[im]; rcp = rel[im]
print(f'bf16-ref vs f32: cos min {c16.min():.6f} mean {c16.mean():.6f} rel-L2 max {r16.max():.4f} mean {r16.mean():.4f}')
print(f'cpp      vs f32: cos min {ccp.min():.6f} mean {ccp.mean():.6f} rel-L2 max {rcp.max():.4f} mean {rcp.mean():.4f}')
print('rank corr of per-row error (cpp vs bf16):', np.corrcoef(rcp.numpy(), r16.numpy())[0,1].round(3))
print('worst image-row norms: ref', e.norm(dim=1)[rcp.argmax()].item(), 'median', e.norm(dim=1).median().item())
ok = ccp.min() >= c16.min() and rcp.mean() <= r16.mean()
print('VISION GOLDEN', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
