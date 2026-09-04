#!/usr/bin/env python3
"""Quantize the fp32 embedding tables left behind by the int4 (MatMulNBits) export.

MatMulNBitsQuantizer only touches MatMul weights. Token embeddings are read by
a Gather, so they stay fp32 and end up as ~65% of every *_q4.onnx file: the
"int4" WebGPU download was larger than the int8 CPU one.

Each large fp32 table feeding a Gather becomes uint8 with a per-row scale and
zero point (255 levels per embedding row, so the error is bounded by that
row's own range rather than the table's), and the Gather is followed by the
dequantization written out in plain ops that every runtime has:

    Gather(table_u8, ids) -> Cast(float) -> Sub(Gather(zp, ids)) -> Mul(Gather(scale, ids))

The scale and zero-point tables are shaped [V, 1] so the gathered values
broadcast over the embedding dimension without an Unsqueeze (whose signature
differs between the opset 11 and 13+ graphs these models use).

    python3 scripts/quantize-embeddings.py in.onnx out.onnx [--min-elements N]

Walks If/Loop subgraphs (the merged decoder is one big If) so a table shared by
both branches is quantized once.
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 1_000_000  # only big tables; leaves tiny lookups (position bias) alone


def quantize_rows(w: np.ndarray):
    """Per-row asymmetric uint8. Returns q [V, d] uint8, scale [V] f32, zp [V] f32."""
    lo = np.minimum(w.min(axis=1), 0.0)
    hi = np.maximum(w.max(axis=1), 0.0)
    scale = (hi - lo) / 255.0
    scale[scale == 0] = 1.0
    zp = np.clip(np.round(-lo / scale), 0, 255)
    q = np.clip(np.round(w / scale[:, None]) + zp[:, None], 0, 255).astype(np.uint8)
    return q, scale.astype(np.float32)[:, None], zp.astype(np.float32)[:, None]


def iter_graphs(graph):
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from iter_graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    yield from iter_graphs(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-elements", type=int, default=MIN_ELEMENTS)
    args = ap.parse_args()

    model = onnx.load(args.src, load_external_data=False)
    top = model.graph
    inits = {t.name: t for t in top.initializer}

    # Which fp32 initializers feed a Gather (anywhere in the graph tree), and are big?
    targets = {}
    for g in iter_graphs(top):
        for node in g.node:
            if node.op_type != "Gather" or not node.input:
                continue
            t = inits.get(node.input[0])
            if t is None or t.data_type != TensorProto.FLOAT or len(t.dims) != 2:
                continue
            axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
            if axis != 0:
                continue
            if int(np.prod(t.dims)) >= args.min_elements:
                targets.setdefault(t.name, []).append((g, node))

    if not targets:
        print("nothing to do: no large fp32 Gather tables", file=sys.stderr)

    saved = 0
    for name, uses in targets.items():
        t = inits[name]
        w = numpy_helper.to_array(t)
        q, scale, zp = quantize_rows(w)
        err = float(np.abs((q.astype(np.float32) - zp) * scale - w).max())
        top.initializer.remove(t)
        top.initializer.extend([
            numpy_helper.from_array(q, name + "_quantized"),
            numpy_helper.from_array(scale, name + "_row_scale"),
            numpy_helper.from_array(zp, name + "_row_zero_point"),
        ])
        for g, node in uses:
            ids = node.input[1]
            node.input[0] = name + "_quantized"
            out = node.output[0]
            node.output[0] = out + "_u8"
            p = f"{out}_deq"
            new_nodes = [
                helper.make_node("Cast", [out + "_u8"], [p + "_f"], to=TensorProto.FLOAT, name=p + "_Cast"),
                helper.make_node("Gather", [name + "_row_zero_point", ids], [p + "_zp"], axis=0, name=p + "_GatherZp"),
                helper.make_node("Sub", [p + "_f", p + "_zp"], [p + "_c"], name=p + "_Sub"),
                helper.make_node("Gather", [name + "_row_scale", ids], [p + "_s"], axis=0, name=p + "_GatherScale"),
                helper.make_node("Mul", [p + "_c", p + "_s"], [out], name=p + "_Mul"),
            ]
            idx = list(g.node).index(node)
            for k, n in enumerate(new_nodes):
                g.node.insert(idx + 1 + k, n)
        saved += w.nbytes - q.nbytes
        print(f"{name}: {list(t.dims)} fp32 -> uint8 per-row, max abs err {err:.3g}, {len(uses)} Gather(s)")

    onnx.save(model, args.dst)
    print(f"{os.path.getsize(args.src)/2**20:.1f} MB -> {os.path.getsize(args.dst)/2**20:.1f} MB "
          f"(embedding bytes saved: {saved/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
