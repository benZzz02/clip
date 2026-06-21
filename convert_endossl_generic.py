"""Convert Flax/MessagePack ViT checkpoint to PyTorch (auto-detect size).

Usage:
    python convert_endossl_generic.py \\
        --input /path/to/checkpoint_0 \\
        --output /path/to/endossl_vitb.pth
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import timm

try:
    import msgpack
except ImportError:
    raise SystemExit("msgpack required. Install with: pip install msgpack")


# ── Flax MessagePack tensor decoder ──────────────────────────────────────

def _decode_shape_and_dtype(shape_bytes):
    pos = 0
    if shape_bytes[pos] == 0x91:
        ndim = 1; pos += 1
    elif shape_bytes[pos] == 0x92:
        ndim = 2; pos += 1
    elif shape_bytes[pos] == 0x93:
        ndim = 3; pos += 1
    elif shape_bytes[pos] == 0x94:
        ndim = 4; pos += 1
    else:
        raise ValueError(f"Unexpected shape array tag 0x{shape_bytes[pos]:02x}")

    dims = []
    for _ in range(ndim):
        b = shape_bytes[pos]
        if b < 0x80:
            dims.append(b); pos += 1
        elif b == 0xCC:
            dims.append(shape_bytes[pos + 1]); pos += 2
        elif b == 0xCD:
            dims.append(int.from_bytes(shape_bytes[pos + 1:pos + 3], "big")); pos += 3
        elif b == 0xCE:
            dims.append(int.from_bytes(shape_bytes[pos + 1:pos + 5], "big")); pos += 5
        elif b == 0xD0:
            v = shape_bytes[pos + 1]
            dims.append(v - 256 if v > 127 else v); pos += 2
        else:
            raise ValueError(f"Unexpected dim tag 0x{b:02x}")

    dtype_tag = shape_bytes[pos]
    if 0xA0 <= dtype_tag < 0xC0:
        strlen = dtype_tag & 0x1F
        dtype = shape_bytes[pos + 1:pos + 1 + strlen].decode("ascii")
        pos += 1 + strlen
    elif dtype_tag == 0xD9:
        strlen = shape_bytes[pos + 1]
        dtype = shape_bytes[pos + 2:pos + 2 + strlen].decode("ascii")
        pos += 2 + strlen
    else:
        raise ValueError(f"Unexpected dtype tag 0x{dtype_tag:02x}")

    return tuple(dims), dtype, pos


def _read_data_size(data_bytes, offset):
    tag = data_bytes[offset]
    if tag == 0xC4:
        return data_bytes[offset + 1], offset + 2
    elif tag == 0xC5:
        return int.from_bytes(data_bytes[offset + 1:offset + 3], "big"), offset + 3
    elif tag == 0xC6:
        return int.from_bytes(data_bytes[offset + 1:offset + 5], "big"), offset + 5
    else:
        raise ValueError(f"Unexpected bin tag 0x{tag:02x}")


def extract_flax_tensor(ext_data):
    pos = 0
    if ext_data[pos] != 0x93:
        raise ValueError(f"Expected fixarray(3) tag, got 0x{ext_data[pos]:02x}")
    shape, dtype, next_pos = _decode_shape_and_dtype(ext_data[pos + 1:])
    data_offset = pos + 1 + next_pos
    data_size, data_start = _read_data_size(ext_data, data_offset)
    raw = ext_data[data_start:data_start + data_size]
    expected = int(np.prod(shape))
    actual = len(raw) // 4
    if expected != actual:
        raise ValueError(f"Shape {shape} expects {expected} elements, got {actual}")
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def extract_teacher_weights(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    teacher_weights = data["teacher_weights"]

    result = OrderedDict()
    def _extract(d, prefix=""):
        if not isinstance(d, dict):
            return
        for key, value in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if hasattr(value, "code") and value.code == 1:
                arr = extract_flax_tensor(value.data)
                result[full_key] = arr
            elif isinstance(value, dict):
                _extract(value, full_key)

    _extract(teacher_weights)
    for key, arr in sorted(result.items()):
        print(f"  {key:<60s} shape={tuple(arr.shape)}")
    return result


def detect_model_size(flax_weights):
    """Auto-detect ViT dimensions from checkpoint weights."""
    # Number of blocks: count encoderblock_N keys
    block_keys = [k for k in flax_weights if "encoderblock_" in k]
    n_blocks = 0
    for k in block_keys:
        parts = k.split("/")[0]
        idx = int(parts.split("_")[1])
        n_blocks = max(n_blocks, idx + 1)

    # Embed dim: from query kernel shape [dim, num_heads, head_dim] or cls_token
    cls = flax_weights.get("ToTokenSequence_0/cls")
    embed_dim = cls.shape[-1] if cls is not None else 768

    # Num heads + head dim: from query kernel [dim, num_heads, head_dim]
    sample_qk = None
    for k in flax_weights:
        if "query/kernel" in k:
            sample_qk = flax_weights[k]
            break
    if sample_qk is not None:
        num_heads = sample_qk.shape[1]
        head_dim = sample_qk.shape[2]
    else:
        num_heads = 12
        head_dim = embed_dim // num_heads

    # Patch size: from patch_embed kernel [KH, KW, C_in, C_out]
    emb = flax_weights.get("ToTokenSequence_0/embedding/kernel")
    patch_size = emb.shape[0] if emb is not None else 16

    # Grid: from pos_embed [1, H, W, dim]
    pos = flax_weights.get("ToTokenSequence_0/posembed_input")
    grid_h = pos.shape[1] if pos is not None else 14

    mlp_ratio = 4

    print(f"\nDetected model size:")
    print(f"  embed_dim={embed_dim}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"  num_blocks={n_blocks}, patch_size={patch_size}, grid={grid_h}x{grid_h}")
    print(f"  mlp_hidden={embed_dim * mlp_ratio}")

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_blocks": n_blocks,
        "patch_size": patch_size,
        "grid_h": grid_h,
        "mlp_ratio": mlp_ratio,
    }


def convert_to_pytorch(flax_weights, cfg):
    embed_dim = cfg["embed_dim"]
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    num_blocks = cfg["num_blocks"]
    num_patches = cfg["grid_h"] * cfg["grid_h"]
    mlp_hidden = embed_dim * cfg["mlp_ratio"]

    pt = OrderedDict()
    stats = {"mapped": 0, "skipped": 0}

    # cls_token
    cls_arr = flax_weights.get("ToTokenSequence_0/cls")
    if cls_arr is not None:
        pt["cls_token"] = torch.from_numpy(cls_arr).clone()
        stats["mapped"] += 1

    # pos_embed
    pos_arr = flax_weights.get("ToTokenSequence_0/posembed_input")
    if pos_arr is not None:
        pos = torch.from_numpy(pos_arr).clone()  # [1, H, W, D]
        pos = pos.reshape(1, num_patches, embed_dim)
        cls_pos = torch.zeros(1, 1, embed_dim, dtype=pos.dtype)
        pt["pos_embed"] = torch.cat([cls_pos, pos], dim=1)  # [1, N+1, D]
        stats["mapped"] += 1
        print(f"  pos_embed: shape={tuple(pt['pos_embed'].shape)}")

    # patch_embed
    emb_kernel = flax_weights.get("ToTokenSequence_0/embedding/kernel")
    emb_bias = flax_weights.get("ToTokenSequence_0/embedding/bias")
    if emb_kernel is not None:
        k = torch.from_numpy(emb_kernel).clone()
        pt["patch_embed.proj.weight"] = k.permute(3, 2, 0, 1).contiguous()
        stats["mapped"] += 1
    if emb_bias is not None:
        pt["patch_embed.proj.bias"] = torch.from_numpy(emb_bias).clone()
        stats["mapped"] += 1

    # blocks
    for i in range(num_blocks):
        prefix = f"blocks.{i}"
        block_key = f"encoderblock_{i}"

        # LayerNorms
        for ln_suffix, pt_ln in [("LayerNorm_0", "norm1"), ("LayerNorm_1", "norm2")]:
            for param, pt_param in [("scale", "weight"), ("bias", "bias")]:
                key = f"{block_key}/{ln_suffix}/{param}"
                arr = flax_weights.get(key)
                if arr is not None:
                    pt[f"{prefix}.{pt_ln}.{pt_param}"] = torch.from_numpy(arr).clone()
                    stats["mapped"] += 1

        # QKV attention
        attn_key = f"{block_key}/MultiHeadDotProductAttention_0"
        qkv_parts = []
        qkv_bias_parts = []
        for qtype in ("query", "key", "value"):
            k = flax_weights.get(f"{attn_key}/{qtype}/kernel")
            b = flax_weights.get(f"{attn_key}/{qtype}/bias")
            if k is not None:
                k_t = torch.from_numpy(k).clone()
                k_t = k_t.reshape(embed_dim, num_heads * head_dim)
                qkv_parts.append(k_t)
                stats["mapped"] += 1
            if b is not None:
                b_t = torch.from_numpy(b).clone().reshape(num_heads * head_dim)
                qkv_bias_parts.append(b_t)
                stats["mapped"] += 1

        if len(qkv_parts) == 3:
            pt[f"{prefix}.attn.qkv.weight"] = torch.cat(qkv_parts, dim=0)
        if len(qkv_bias_parts) == 3:
            pt[f"{prefix}.attn.qkv.bias"] = torch.cat(qkv_bias_parts, dim=0)

        # Attention projection out
        out_k = flax_weights.get(f"{attn_key}/out/kernel")
        out_b = flax_weights.get(f"{attn_key}/out/bias")
        if out_k is not None:
            k_t = torch.from_numpy(out_k).clone().reshape(embed_dim, embed_dim)
            pt[f"{prefix}.attn.proj.weight"] = k_t
            stats["mapped"] += 1
        if out_b is not None:
            pt[f"{prefix}.attn.proj.bias"] = torch.from_numpy(out_b).clone()
            stats["mapped"] += 1

        # MLP
        mlp_key = f"{block_key}/MlpBlock_0"
        fc1_k = flax_weights.get(f"{mlp_key}/Dense_0/kernel")
        fc1_b = flax_weights.get(f"{mlp_key}/Dense_0/bias")
        if fc1_k is not None:
            pt[f"{prefix}.mlp.fc1.weight"] = torch.from_numpy(fc1_k).clone().t().contiguous()
            stats["mapped"] += 1
        if fc1_b is not None:
            pt[f"{prefix}.mlp.fc1.bias"] = torch.from_numpy(fc1_b).clone()
            stats["mapped"] += 1

        fc2_k = flax_weights.get(f"{mlp_key}/Dense_1/kernel")
        fc2_b = flax_weights.get(f"{mlp_key}/Dense_1/bias")
        if fc2_k is not None:
            pt[f"{prefix}.mlp.fc2.weight"] = torch.from_numpy(fc2_k).clone().t().contiguous()
            stats["mapped"] += 1
        if fc2_b is not None:
            pt[f"{prefix}.mlp.fc2.bias"] = torch.from_numpy(fc2_b).clone()
            stats["mapped"] += 1

    # encoder_norm
    for param, pt_param in [("scale", "weight"), ("bias", "bias")]:
        arr = flax_weights.get(f"encoder_norm/{param}")
        if arr is not None:
            pt[f"norm.{pt_param}"] = torch.from_numpy(arr).clone()
            stats["mapped"] += 1

    print(f"\nConverted: {stats['mapped']} params mapped, {stats['skipped']} skipped")
    return pt, stats


def validate_and_save(state_dict, cfg, output_path):
    target = timm.create_model(
        f"vit_{cfg['patch_size']}x{cfg['grid_h'] * cfg['patch_size']}",
        pretrained=False,
        num_classes=0,
        global_pool="token",
    )
    # Try different timm model names
    if cfg["embed_dim"] == 768 and cfg["num_blocks"] == 12 and cfg["num_heads"] == 12:
        model_name = "vit_base_patch16_224"
    elif cfg["embed_dim"] == 384 and cfg["num_blocks"] == 12 and cfg["num_heads"] == 6:
        model_name = "vit_small_patch16_224"
    elif cfg["embed_dim"] == 1024 and cfg["num_blocks"] == 24 and cfg["num_heads"] == 16:
        model_name = "vit_large_patch16_224"
    else:
        model_name = f"vit_custom_p{cfg['patch_size']}_d{cfg['embed_dim']}"

    target = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="token")
    target_keys = set(target.state_dict().keys())
    mapped_keys = set(state_dict.keys())
    missing = target_keys - mapped_keys
    extra = mapped_keys - target_keys

    if missing:
        print(f"\nMissing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"  {k}")
    if extra:
        print(f"\nWARNING: Extra keys ({len(extra)}):")
        for k in sorted(extra):
            print(f"  {k}")

    # Shape check
    mismatches = []
    for k in sorted(target_keys & mapped_keys):
        if state_dict[k].shape != target.state_dict()[k].shape:
            mismatches.append(f"  {k}: {tuple(state_dict[k].shape)} vs {tuple(target.state_dict()[k].shape)}")
    if mismatches:
        print(f"\nShape mismatches ({len(mismatches)}):")
        for m in mismatches:
            print(m)
    else:
        print("All shapes match. OK")
        target.load_state_dict(state_dict, strict=True)
        print("Strict load test passed.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(state_dict, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSaved {len(state_dict)} keys ({size_mb:.1f} MB) to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert EndoSSL Flax checkpoint to PyTorch (auto-detect size)")
    parser.add_argument("--input", type=str, required=True, help="Path to Flax checkpoint")
    parser.add_argument("--output", type=str, default="endossl_vit.pth", help="Output PyTorch .pth path")
    args = parser.parse_args()

    flax_weights = extract_teacher_weights(args.input)
    cfg = detect_model_size(flax_weights)
    pt_state, stats = convert_to_pytorch(flax_weights, cfg)
    validate_and_save(pt_state, cfg, args.output)


if __name__ == "__main__":
    main()
