"""Convert EndoSSL Flax/MessagePack checkpoint to PyTorch ViT-L state_dict.

The EndoSSL Google Drive checkpoint is a Flax MessagePack file containing teacher
weights from MSN pretraining on private laparoscopic videos.

Usage:
    python convert_endossl_to_torch.py \
        --input D:/checkpoint_0 \
        --output checkpoints/endossl_vitl.pth

Requires: msgpack (pip install msgpack)
"""

import argparse
import os
import struct
from collections import OrderedDict

import numpy as np

try:
    import msgpack
except ImportError:
    raise SystemExit("msgpack required. Install with: pip install msgpack")

try:
    import torch
    import timm
except ImportError:
    raise SystemExit("torch and timm required. Install with: pip install torch timm")


# ── Flax MessagePack tensor decoder ──────────────────────────────────────

def _decode_shape_and_dtype(shape_bytes):
    """Decode the shape array from a Flax-serialized MessagePack entry.

    The shape is encoded as a nested MessagePack array. Returns (shape_tuple, dtype_str, rest_offset).
    """
    # Read the outer array: [shape_array, dtype_string]
    pos = 0

    # Parse the shape array
    if shape_bytes[pos] == 0x91:  # fixarray(1)
        ndim = 1
        pos += 1
    elif shape_bytes[pos] == 0x92:  # fixarray(2)
        ndim = 2
        pos += 1
    elif shape_bytes[pos] == 0x93:  # fixarray(3)
        ndim = 3
        pos += 1
    elif shape_bytes[pos] == 0x94:  # fixarray(4)
        ndim = 4
        pos += 1
    else:
        raise ValueError(f"Unexpected shape array tag 0x{shape_bytes[pos]:02x} at offset 0")

    dims = []
    for _ in range(ndim):
        b = shape_bytes[pos]
        if b < 0x80:  # fixint
            dims.append(b)
            pos += 1
        elif b == 0xCC:  # uint8
            dims.append(shape_bytes[pos + 1])
            pos += 2
        elif b == 0xCD:  # uint16
            dims.append(int.from_bytes(shape_bytes[pos + 1 : pos + 3], "big"))
            pos += 3
        elif b == 0xCE:  # uint32
            dims.append(int.from_bytes(shape_bytes[pos + 1 : pos + 5], "big"))
            pos += 5
        elif b == 0xD0:  # int8
            dims.append(shape_bytes[pos + 1] - 256 if shape_bytes[pos + 1] > 127 else shape_bytes[pos + 1])
            pos += 2
        else:
            raise ValueError(f"Unexpected dim tag 0x{b:02x} at offset {pos}")

    # Parse dtype string (fixstr or str8/16)
    dtype_tag = shape_bytes[pos]
    if 0xA0 <= dtype_tag < 0xC0:
        strlen = dtype_tag & 0x1F
        dtype = shape_bytes[pos + 1 : pos + 1 + strlen].decode("ascii")
        pos += 1 + strlen
    elif dtype_tag == 0xD9:  # str8
        strlen = shape_bytes[pos + 1]
        dtype = shape_bytes[pos + 2 : pos + 2 + strlen].decode("ascii")
        pos += 2 + strlen
    else:
        raise ValueError(f"Unexpected dtype tag 0x{dtype_tag:02x} at offset {pos}")

    return tuple(dims), dtype, pos


def _read_data_size(data_bytes, offset):
    """Read bin data size from MessagePack tag at offset."""
    tag = data_bytes[offset]
    if tag == 0xC4:  # bin8
        return data_bytes[offset + 1], offset + 2
    elif tag == 0xC5:  # bin16
        return int.from_bytes(data_bytes[offset + 1 : offset + 3], "big"), offset + 3
    elif tag == 0xC6:  # bin32
        return int.from_bytes(data_bytes[offset + 1 : offset + 5], "big"), offset + 5
    else:
        raise ValueError(f"Unexpected bin tag 0x{tag:02x} at offset {offset}")


def extract_flax_tensor(ext_data):
    """Extract a numpy array from a Flax MessagePack ExtType(code=1) data blob.

    Format: outer array [shape_array, dtype_string, raw_float32_bytes]
    """
    # Outer array tag
    pos = 0
    if ext_data[pos] != 0x93:  # fixarray(3)
        raise ValueError(f"Expected fixarray(3) tag, got 0x{ext_data[pos]:02x}")

    # Parse shape and dtype from next bytes
    shape, dtype, next_pos = _decode_shape_and_dtype(ext_data[pos + 1 :])

    # Read raw data
    data_offset = pos + 1 + next_pos
    data_size, data_start = _read_data_size(ext_data, data_offset)
    raw = ext_data[data_start : data_start + data_size]

    expected = int(np.prod(shape))
    actual = len(raw) // 4  # float32
    if expected != actual:
        raise ValueError(f"Shape {shape} expects {expected} elements, got {actual}")

    arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    return arr


# ── Conversion ───────────────────────────────────────────────────────────


def extract_teacher_weights(checkpoint_path):
    """Load the Flax MessagePack checkpoint and return teacher_weights dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        data = msgpack.unpack(f, raw=False)

    teacher_weights = data["teacher_weights"]
    print(f"teacher_weights: {len(teacher_weights)} keys")

    def _extract_recursive(d, prefix=""):
        """Recursively extract all ExtType tensors from a nested dict."""
        if not isinstance(d, dict):
            return
        for key, value in d.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if hasattr(value, "code") and value.code == 1:
                arr = extract_flax_tensor(value.data)
                result[full_key] = arr
            elif isinstance(value, dict):
                _extract_recursive(value, full_key)

    result = OrderedDict()
    _extract_recursive(teacher_weights)

    for key, arr in sorted(result.items()):
        print(f"  {key:<60s} shape={tuple(arr.shape)}")

    return result


def convert_to_pytorch(flax_weights):
    """Convert Flax teacher_weights to PyTorch ViT-L state_dict."""
    pt = OrderedDict()
    stats = {"mapped": 0, "skipped": 0}

    # ── cls_token ──────────────────────────────────────────────────────
    cls_arr = flax_weights.get("ToTokenSequence_0/cls")
    if cls_arr is not None:
        # shape [1, 1, 1024] → [1, 1, 1024]
        pt["cls_token"] = torch.from_numpy(cls_arr).clone()
        stats["mapped"] += 1
    else:
        print("WARNING: cls_token not found")

    # ── pos_embed ──────────────────────────────────────────────────────
    pos_arr = flax_weights.get("ToTokenSequence_0/posembed_input")
    if pos_arr is not None:
        # Flax: [1, 14, 14, 1024] — patch positions only, no cls position
        # PyTorch timm: [1, 197, 1024] — cls + patch positions
        pos = torch.from_numpy(pos_arr).clone()  # [1, 14, 14, 1024]
        pos = pos.reshape(1, -1, 1024)  # [1, 196, 1024]
        # Prepend cls position (zero-initialized, same as standard ViT)
        cls_pos = torch.zeros(1, 1, 1024, dtype=pos.dtype)
        pt["pos_embed"] = torch.cat([cls_pos, pos], dim=1)  # [1, 197, 1024]
        stats["mapped"] += 1
        print(f"  pos_embed: added cls position → shape={tuple(pt['pos_embed'].shape)}")
    else:
        print("WARNING: pos_embed not found")

    # ── patch_embed ──────────────────────────────────────────────────
    emb_kernel = flax_weights.get("ToTokenSequence_0/embedding/kernel")
    emb_bias = flax_weights.get("ToTokenSequence_0/embedding/bias")
    if emb_kernel is not None:
        # Flax/TF: [KH, KW, C_in, C_out] = [16, 16, 3, 1024]
        # PyTorch: [C_out, C_in, KH, KW] = [1024, 3, 16, 16]
        k = torch.from_numpy(emb_kernel).clone()
        pt["patch_embed.proj.weight"] = k.permute(3, 2, 0, 1).contiguous()  # [1024, 3, 16, 16]
        stats["mapped"] += 1
    if emb_bias is not None:
        pt["patch_embed.proj.bias"] = torch.from_numpy(emb_bias).clone()
        stats["mapped"] += 1

    # ── encoder blocks 0..23 ───────────────────────────────────────────
    n_blocks = 24
    for i in range(n_blocks):
        prefix = f"blocks.{i}"
        block_key = f"encoderblock_{i}"

        # LayerNorm_0 → norm1
        for ln_suffix, pt_ln in [("LayerNorm_0", "norm1"), ("LayerNorm_1", "norm2")]:
            for param, pt_param in [("scale", "weight"), ("bias", "bias")]:
                key = f"{block_key}/{ln_suffix}/{param}"
                arr = flax_weights.get(key)
                if arr is not None:
                    pt[f"{prefix}.{pt_ln}.{pt_param}"] = torch.from_numpy(arr).clone()
                    stats["mapped"] += 1
                else:
                    print(f"WARNING: {key} not found")

        # QKV attention
        # Flax: query/kernel [1024, 16, 64], key/kernel [1024, 16, 64], value/kernel [1024, 16, 64]
        # → PyTorch: qkv.weight [3072, 1024]
        attn_key = f"{block_key}/MultiHeadDotProductAttention_0"
        qkv_parts = []
        qkv_bias_parts = []
        for qtype in ("query", "key", "value"):
            k = flax_weights.get(f"{attn_key}/{qtype}/kernel")
            b = flax_weights.get(f"{attn_key}/{qtype}/bias")
            if k is not None:
                # [1024, 16, 64] → [1024, 1024]
                k_t = torch.from_numpy(k).clone()
                k_t = k_t.reshape(1024, 1024)
                qkv_parts.append(k_t)
                stats["mapped"] += 1
            if b is not None:
                # [16, 64] → [1024]
                b_t = torch.from_numpy(b).clone().reshape(1024)
                qkv_bias_parts.append(b_t)
                stats["mapped"] += 1

        if len(qkv_parts) == 3:
            pt[f"{prefix}.attn.qkv.weight"] = torch.cat(qkv_parts, dim=0)
        if len(qkv_bias_parts) == 3:
            pt[f"{prefix}.attn.qkv.bias"] = torch.cat(qkv_bias_parts, dim=0)

        # Attention projection (out)
        out_k = flax_weights.get(f"{attn_key}/out/kernel")
        out_b = flax_weights.get(f"{attn_key}/out/bias")
        if out_k is not None:
            # [16, 64, 1024] → [1024, 1024]
            k_t = torch.from_numpy(out_k).clone().reshape(1024, 1024)
            pt[f"{prefix}.attn.proj.weight"] = k_t
            stats["mapped"] += 1
        if out_b is not None:
            pt[f"{prefix}.attn.proj.bias"] = torch.from_numpy(out_b).clone()
            stats["mapped"] += 1

        # MLP
        mlp_key = f"{block_key}/MlpBlock_0"
        # Dense_0: [1024, 4096] → mlp.fc1.weight (PyTorch: [4096, 1024])
        fc1_k = flax_weights.get(f"{mlp_key}/Dense_0/kernel")
        fc1_b = flax_weights.get(f"{mlp_key}/Dense_0/bias")
        if fc1_k is not None:
            pt[f"{prefix}.mlp.fc1.weight"] = torch.from_numpy(fc1_k).clone().t().contiguous()
            stats["mapped"] += 1
        if fc1_b is not None:
            pt[f"{prefix}.mlp.fc1.bias"] = torch.from_numpy(fc1_b).clone()
            stats["mapped"] += 1

        # Dense_1: [4096, 1024] → mlp.fc2.weight (PyTorch: [1024, 4096])
        fc2_k = flax_weights.get(f"{mlp_key}/Dense_1/kernel")
        fc2_b = flax_weights.get(f"{mlp_key}/Dense_1/bias")
        if fc2_k is not None:
            pt[f"{prefix}.mlp.fc2.weight"] = torch.from_numpy(fc2_k).clone().t().contiguous()
            stats["mapped"] += 1
        if fc2_b is not None:
            pt[f"{prefix}.mlp.fc2.bias"] = torch.from_numpy(fc2_b).clone()
            stats["mapped"] += 1

    # ── encoder_norm → norm ──────────────────────────────────────────
    for param, pt_param in [("scale", "weight"), ("bias", "bias")]:
        arr = flax_weights.get(f"encoder_norm/{param}")
        if arr is not None:
            pt[f"norm.{pt_param}"] = torch.from_numpy(arr).clone()
            stats["mapped"] += 1

    print(f"\nConverted: {stats['mapped']} params mapped, {stats['skipped']} skipped")
    return pt, stats


def validate_state_dict(state_dict):
    """Check state_dict against timm ViT-L target."""
    print("\nValidating against timm vit_large_patch16_224 target...")
    target = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="token",
    )
    target_keys = set(target.state_dict().keys())
    mapped_keys = set(state_dict.keys())

    missing = target_keys - mapped_keys
    extra = mapped_keys - target_keys

    if missing:
        print(f"\nERROR: Missing keys ({len(missing)}):")
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
            mismatches.append(
                f"  {k}: checkpoint {tuple(state_dict[k].shape)} vs target {tuple(target.state_dict()[k].shape)}"
            )
    if mismatches:
        print(f"\nERROR: Shape mismatches ({len(mismatches)}):")
        for m in mismatches:
            print(m)
    else:
        print("All shapes match. OK")

    return len(missing) == 0 and len(mismatches) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert EndoSSL Flax/MessagePack checkpoint to PyTorch ViT-L state_dict"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Flax checkpoint file (e.g. D:/checkpoint_0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/endossl_vitl.pth",
        help="Output path for the PyTorch .pth file",
    )
    args = parser.parse_args()

    # 1. Extract weights from Flax checkpoint
    flax_weights = extract_teacher_weights(args.input)

    # 2. Convert to PyTorch format
    pt_state, stats = convert_to_pytorch(flax_weights)

    # 3. Validate
    ok = validate_state_dict(pt_state)
    if not ok:
        print("\nValidation failed. Check the errors above.")
        raise SystemExit(1)

    # 4. Test load
    print("\nRunning strict load test...")
    target = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="token",
    )
    target.load_state_dict(pt_state, strict=True)
    print("Strict load test passed. OK")

    # 5. Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(pt_state, args.output)
    file_size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved {len(pt_state)} keys ({file_size_mb:.1f} MB) to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
