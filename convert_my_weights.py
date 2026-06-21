"""Convert EndoSSL Flax checkpoint to PyTorch (auto-detect size, no argparse)."""
import sys, os, msgpack, numpy as np, torch, timm
from collections import OrderedDict

ckpt_path = sys.argv[1]
out_path = sys.argv[2]

print(f"Loading: {ckpt_path}")
with open(ckpt_path, "rb") as f:
    data = msgpack.unpack(f, raw=False)
cls_weights = data["cls_weights"] if "cls_weights" in data else data["teacher_weights"]

# Decode all tensors
def decode(d, prefix="", res=None):
    if res is None: res = OrderedDict()
    if not isinstance(d, dict): return res
    for k, v in d.items():
        fk = f"{prefix}/{k}" if prefix else k
        if hasattr(v, "code") and v.code == 1:
            try:
                from convert_endossl_to_torch import extract_flax_tensor
                arr = extract_flax_tensor(v.data)
                res[fk] = arr
            except:
                print(f"  SKIP {fk}")
        elif isinstance(v, dict):
            decode(v, fk, res)
    return res

weights = decode(cls_weights)
print(f"  Extracted {len(weights)} tensors")

# Detect size
embed_dim = weights["ToTokenSequence_0/cls"].shape[-1]
qk = [v for k,v in weights.items() if "query/kernel" in k][0]
num_heads = qk.shape[1]; head_dim = qk.shape[2]
n_blocks = max(int(k.split("_")[1].split("/")[0]) for k in weights if "encoderblock_" in k) + 1
print(f"  ViT: dim={embed_dim}, heads={num_heads}, hdim={head_dim}, blocks={n_blocks}")

# Convert
pt = OrderedDict()
pt["cls_token"] = torch.from_numpy(weights["ToTokenSequence_0/cls"]).clone()
pos = torch.from_numpy(weights["ToTokenSequence_0/posembed_input"]).clone()
pos = pos.reshape(1, -1, embed_dim)
pt["pos_embed"] = torch.cat([torch.zeros(1, 1, embed_dim), pos], dim=1)
k = torch.from_numpy(weights["ToTokenSequence_0/embedding/kernel"]).clone()
pt["patch_embed.proj.weight"] = k.permute(3, 2, 0, 1).contiguous()
pt["patch_embed.proj.bias"] = torch.from_numpy(weights["ToTokenSequence_0/embedding/bias"]).clone()

for i in range(n_blocks):
    pfx = f"blocks.{i}"; ek = f"encoderblock_{i}"
    for ln, pl in [("LayerNorm_0","norm1"),("LayerNorm_1","norm2")]:
        for ps, pp in [("scale","weight"),("bias","bias")]:
            key = f"{ek}/{ln}/{ps}"
            if key in weights: pt[f"{pfx}.{pl}.{pp}"] = torch.from_numpy(weights[key]).clone()
    at = f"{ek}/MultiHeadDotProductAttention_0"
    qp, qb = [], []
    for qt in ("query","key","value"):
        if f"{at}/{qt}/kernel" in weights:
            qp.append(torch.from_numpy(weights[f"{at}/{qt}/kernel"]).clone().reshape(embed_dim, num_heads*head_dim))
        if f"{at}/{qt}/bias" in weights:
            qb.append(torch.from_numpy(weights[f"{at}/{qt}/bias"]).clone().reshape(num_heads*head_dim))
    if len(qp)==3:
        pt[f"{pfx}.attn.qkv.weight"] = torch.cat(qp, dim=0)
        pt[f"{pfx}.attn.qkv.bias"] = torch.cat(qb, dim=0)
    if f"{at}/out/kernel" in weights:
        pt[f"{pfx}.attn.proj.weight"] = torch.from_numpy(weights[f"{at}/out/kernel"]).clone().reshape(embed_dim, embed_dim)
    if f"{at}/out/bias" in weights:
        pt[f"{pfx}.attn.proj.bias"] = torch.from_numpy(weights[f"{at}/out/bias"]).clone()
    mlp = f"{ek}/MlpBlock_0"
    for s, p in [("Dense_0","fc1"),("Dense_1","fc2")]:
        if f"{mlp}/{s}/kernel" in weights:
            pt[f"{pfx}.mlp.{p}.weight"] = torch.from_numpy(weights[f"{mlp}/{s}/kernel"]).clone().t().contiguous()
        if f"{mlp}/{s}/bias" in weights:
            pt[f"{pfx}.mlp.{p}.bias"] = torch.from_numpy(weights[f"{mlp}/{s}/bias"]).clone()
for ps, pp in [("scale","weight"),("bias","bias")]:
    if f"encoder_norm/{ps}" in weights:
        pt[f"norm.{pp}"] = torch.from_numpy(weights[f"encoder_norm/{ps}"]).clone()

# Validate
names = {12: "vit_base_patch16_224", 24: "vit_large_patch16_224"}
mn = names.get(n_blocks, "vit_base_patch16_224")
target = timm.create_model(mn, pretrained=False, num_classes=0, global_pool="token")
msg = target.load_state_dict(pt, strict=True)
if msg.missing_keys or msg.unexpected_keys:
    print(f"WARNING: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
else:
    print("  Strict validation: OK")

torch.save(pt, out_path)
print(f"  Saved {len(pt)} keys ({os.path.getsize(out_path)/1024/1024:.0f}MB) to {out_path}")

# Also convert the other checkpoint if it exists
other = os.path.join(os.path.dirname(ckpt_path), "checkpoint_0")
if os.path.isfile(other) and ckpt_path != other:
    out2 = os.path.join(os.path.dirname(out_path), "endossl_vits.pth")
    print(f"\nConverting: {other} → {out2}")
    with open(other, "rb") as f:
        data2 = msgpack.unpack(f, raw=False)
    cls2 = data2["cls_weights"] if "cls_weights" in data2 else data2["teacher_weights"]
    w2 = decode(cls2)
    embed_dim2 = w2["ToTokenSequence_0/cls"].shape[-1]
    qk2 = [v for k,v in w2.items() if "query/kernel" in k][0]
    nh2 = qk2.shape[1]; hd2 = qk2.shape[2]
    nb2 = max(int(k.split("_")[1].split("/")[0]) for k in w2 if "encoderblock_" in k) + 1
    print(f"  ViT: dim={embed_dim2}, heads={nh2}, hdim={hd2}, blocks={nb2}")
    pt2 = OrderedDict()
    pt2["cls_token"] = torch.from_numpy(w2["ToTokenSequence_0/cls"]).clone()
    pos2 = torch.from_numpy(w2["ToTokenSequence_0/posembed_input"]).clone()
    pos2 = pos2.reshape(1, -1, embed_dim2)
    pt2["pos_embed"] = torch.cat([torch.zeros(1, 1, embed_dim2), pos2], dim=1)
    k2 = torch.from_numpy(w2["ToTokenSequence_0/embedding/kernel"]).clone()
    pt2["patch_embed.proj.weight"] = k2.permute(3, 2, 0, 1).contiguous()
    pt2["patch_embed.proj.bias"] = torch.from_numpy(w2["ToTokenSequence_0/embedding/bias"]).clone()
    for i in range(nb2):
        pfx = f"blocks.{i}"; ek = f"encoderblock_{i}"
        for ln, pl in [("LayerNorm_0","norm1"),("LayerNorm_1","norm2")]:
            for ps, pp in [("scale","weight"),("bias","bias")]:
                kk = f"{ek}/{ln}/{ps}"
                if kk in w2: pt2[f"{pfx}.{pl}.{pp}"] = torch.from_numpy(w2[kk]).clone()
        at = f"{ek}/MultiHeadDotProductAttention_0"
        qp2, qb2 = [], []
        for qt in ("query","key","value"):
            if f"{at}/{qt}/kernel" in w2:
                qp2.append(torch.from_numpy(w2[f"{at}/{qt}/kernel"]).clone().reshape(embed_dim2, nh2*hd2))
            if f"{at}/{qt}/bias" in w2:
                qb2.append(torch.from_numpy(w2[f"{at}/{qt}/bias"]).clone().reshape(nh2*hd2))
        if len(qp2)==3:
            pt2[f"{pfx}.attn.qkv.weight"] = torch.cat(qp2, dim=0)
            pt2[f"{pfx}.attn.qkv.bias"] = torch.cat(qb2, dim=0)
        if f"{at}/out/kernel" in w2:
            pt2[f"{pfx}.attn.proj.weight"] = torch.from_numpy(w2[f"{at}/out/kernel"]).clone().reshape(embed_dim2, embed_dim2)
        if f"{at}/out/bias" in w2:
            pt2[f"{pfx}.attn.proj.bias"] = torch.from_numpy(w2[f"{at}/out/bias"]).clone()
        mlp = f"{ek}/MlpBlock_0"
        for s, p in [("Dense_0","fc1"),("Dense_1","fc2")]:
            if f"{mlp}/{s}/kernel" in w2:
                pt2[f"{pfx}.mlp.{p}.weight"] = torch.from_numpy(w2[f"{mlp}/{s}/kernel"]).clone().t().contiguous()
            if f"{mlp}/{s}/bias" in w2:
                pt2[f"{pfx}.mlp.{p}.bias"] = torch.from_numpy(w2[f"{mlp}/{s}/bias"]).clone()
    for ps, pp in [("scale","weight"),("bias","bias")]:
        if f"encoder_norm/{ps}" in w2:
            pt2[f"norm.{pp}"] = torch.from_numpy(w2[f"encoder_norm/{ps}"]).clone()
    torch.save(pt2, out2)
    print(f"  Saved {len(pt2)} keys ({os.path.getsize(out2)/1024/1024:.0f}MB) to {out2}")
