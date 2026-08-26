#!/usr/bin/env python3
"""
Sanity check: the manually split FULL path used by the semantic evaluator must
match the repository's original STREAM.forward_test() before any adaptive logic
is trusted.
"""
import argparse
from collections import deque

import numpy as np
import torch

from pcdet.models import load_data_to_gpu
from single_roi_common import (
    add_base_args,
    build_env,
    cuda_autocast,
    extract_full_bev,
    scalar_string,
)


def parse_args():
    p = argparse.ArgumentParser('Check manual full-path equivalence')
    add_base_args(p, training=False)
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--tol', type=float, default=1e-5)
    return p.parse_args()


def compare_pred(a, b, tol):
    assert len(a) == len(b) == 1
    a, b = a[0], b[0]
    for key in ['pred_boxes', 'pred_scores', 'pred_labels']:
        assert key in a and key in b, key
    assert a['pred_boxes'].shape == b['pred_boxes'].shape, (a['pred_boxes'].shape, b['pred_boxes'].shape)
    assert a['pred_scores'].shape == b['pred_scores'].shape
    assert torch.equal(a['pred_labels'], b['pred_labels']), 'pred_labels mismatch'
    box_diff = float((a['pred_boxes'].float() - b['pred_boxes'].float()).abs().max().item()) if a['pred_boxes'].numel() else 0.0
    score_diff = float((a['pred_scores'].float() - b['pred_scores'].float()).abs().max().item()) if a['pred_scores'].numel() else 0.0
    assert box_diff <= tol and score_diff <= tol, (box_diff, score_diff, tol)
    return box_diff, score_diff, int(a['pred_boxes'].shape[0])


def main():
    args = parse_args()
    cfg, dataset, loader, model, logger = build_env(args, training=False)
    assert args.batch_size == 1
    assert args.frames > 0

    manual_history = deque(maxlen=len(model.history_tag) if model.history_tag is not None else 1)
    last_scene = None
    amp_enabled = bool(model.use_amp_dict.get('TEST', False))

    # Ensure original STREAM state starts clean.
    if model.history_feature_queue is not None:
        model.history_feature_queue.clear()

    checked = 0
    for bi, batch in enumerate(loader):
        if checked >= args.frames:
            break
        load_data_to_gpu(batch)
        token = batch['token']
        scene = scalar_string(token.get('scene', None))
        prev_sample_idx = scalar_string(token.get('prev_sample_idx', None))
        first = (last_scene is None or scene != last_scene or prev_sample_idx == '')
        if first:
            manual_history.clear()
        last_scene = scene

        # Manual full path: exactly the split used by semantic evaluator.
        cur_data, full_bev = extract_full_bev(model, token)
        cur_data['spatial_features'] = full_bev
        cur_data['history_features'] = manual_history
        current_for_history = full_bev.detach().clone()
        with torch.no_grad(), cuda_autocast(amp_enabled):
            for m in model.fusion_module:
                cur_data = m(cur_data)
            for m in model.after_fusion_blocks:
                cur_data = m(cur_data)
            manual_pred, _ = model.post_processing(cur_data)
        if model.history_tag is not None:
            manual_history.append((cur_data['this_sample_idx'], {'spatial_features': current_for_history}))

        # Original repository path on the same input; it maintains its own queue.
        with torch.no_grad():
            original_pred, _ = model(batch)

        box_diff, score_diff, n = compare_pred(manual_pred, original_pred, args.tol)
        print(f'frame={bi} scene={scene} first={int(first)} n={n} box_diff={box_diff:.3g} score_diff={score_diff:.3g}')
        checked += 1

    assert checked == args.frames, (checked, args.frames)
    print('FULL PIPELINE EQUIVALENCE: PASS')


if __name__ == '__main__':
    main()
