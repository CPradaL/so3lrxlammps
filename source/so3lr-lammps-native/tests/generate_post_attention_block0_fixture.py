#!/usr/bin/env python3
"""Generate an independent PyTorch reference for block-0 post-attention work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def discover_root(wrapper: object) -> torch.nn.Module:
    roots: list[torch.nn.Module] = []
    seen = {id(wrapper)}

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            roots.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for value in vars(wrapper).values():
        visit(value)
    if len(roots) != 1:
        raise RuntimeError(f"expected one root module, found {len(roots)}")
    return roots[0]


def find_so3lr(root: torch.nn.Module) -> torch.nn.Module:
    for module in root.modules():
        if type(module).__name__ == "SO3LR":
            return module
    raise RuntimeError("checkpoint does not contain SO3LR")


def flat(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().contiguous().reshape(-1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    wrapper = torch.load(args.model, map_location="cpu", weights_only=False)
    model = find_so3lr(discover_root(wrapper)).eval()
    block = model.euclidean_transformers[0]
    if not (
        block.layer_normalization_1
        and block.layer_normalization_2
        and block.residual_mlp_1
        and not block.residual_mlp_2
    ):
        raise RuntimeError("unexpected block-0 residual/normalization contract")
    interaction = block.interaction_block
    if interaction.degree_repeats.detach().cpu().tolist() != [3, 5, 7, 9]:
        raise RuntimeError("unexpected interaction degree contract")

    nodes = 7
    node = torch.arange(nodes, dtype=torch.float64)[:, None]
    inv_channel = torch.arange(128, dtype=torch.float64)[None, :]
    ev_channel = torch.arange(24, dtype=torch.float64)[None, :]
    inv_features = (
        0.45 * torch.sin((node + 1.0) * (inv_channel + 0.5) / 31.0)
        + 0.18 * torch.cos((node + 0.25) * (inv_channel + 1.0) / 19.0)
    )
    ev_features = (
        0.22 * torch.sin((node + 0.5) * (ev_channel + 0.75) / 9.0)
        - 0.13 * torch.cos((node + 1.25) * (ev_channel + 1.0) / 7.0)
    )
    d_att_inv = (
        0.09 * torch.cos((node + 0.75) * (inv_channel + 1.5) / 23.0)
        - 0.04 * torch.sin((node + 1.5) * (inv_channel + 0.25) / 17.0)
    )
    d_att_ev = (
        0.07 * torch.cos((node + 0.2) * (ev_channel + 1.25) / 8.0)
        + 0.03 * torch.sin((node + 1.1) * (ev_channel + 0.5) / 6.0)
    )

    with torch.no_grad():
        att_inv = inv_features + d_att_inv
        att_ev = ev_features + d_att_ev
        norm1 = block.layer_norm_inv_1(att_inv)
        post_mlp = norm1 + block.mlp_1(norm1)
        ev_invariants = interaction.so3_conv_invariants(att_ev)
        transformed = interaction.linear_layer(
            torch.concatenate([post_mlp, ev_invariants], dim=-1)
        )
        d_inv = transformed[:, :128]
        b_ev = transformed[:, 128:]
        expanded_b = torch.repeat_interleave(
            b_ev, interaction.degree_repeats, dim=-1, output_size=24
        )
        d_ev = expanded_b * att_ev
        pre_norm2_inv = post_mlp + d_inv
        final_inv = block.layer_norm_inv_2(pre_norm2_inv)
        final_ev = att_ev + d_ev

    fixture = {
        "schema": "so3lr-native-post-attention-block0-fixture-v1",
        "nodes": nodes,
        "degree_repeats": [3, 5, 7, 9],
        "layer_norm_eps": 1.0e-6,
        "inv_features": flat(inv_features),
        "ev_features": flat(ev_features),
        "d_att_inv": flat(d_att_inv),
        "d_att_ev": flat(d_att_ev),
        "attention_residual_inv": flat(att_inv),
        "attention_residual_ev": flat(att_ev),
        "layer_norm_1": flat(norm1),
        "post_mlp_inv": flat(post_mlp),
        "interaction_ev_invariants": flat(ev_invariants),
        "interaction_transformed": flat(transformed),
        "pre_layer_norm_2_inv": flat(pre_norm2_inv),
        "final_inv": flat(final_inv),
        "final_ev": flat(final_ev),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"fixture={args.output}")
    print(f"nodes={nodes}")
    print("layer_normalization_1=1")
    print("residual_mlp_1=1")
    print("interaction_width=132")
    print("residual_mlp_2=0")
    print("layer_normalization_2=1")
    print("pytorch_post_attention_block0_fixture=PASS")


if __name__ == "__main__":
    main()
