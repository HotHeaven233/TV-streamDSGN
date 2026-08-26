#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from pcdet.models import load_data_to_gpu
from pcdet.models.adaptive_single_roi import (
    BEVCenterPredictor,
    BEVTemporalPredictor,
    oracle_center_from_error,
    gaussian_heatmap,
    box_to_mask,
    center_to_box,
)

from single_roi_common import add_base_args, build_env, cuda_autocast, extract_full_bev


def parse_args():
    p = argparse.ArgumentParser('Train BEV-only predictor and single-ROI center head')
    add_base_args(p, training=True)
    p.add_argument('--roi-h', type=int, default=96)
    p.add_argument('--roi-w', type=int, default=96)
    p.add_argument('--sigma', type=float, default=8.0)
    p.add_argument('--age-cap', type=int, default=8)
    p.add_argument('--stage', choices=['predictor', 'center', 'joint'], default='joint')
    p.add_argument('--lambda-heat', type=float, default=5.0)
    p.add_argument('--lambda-pred', type=float, default=1.0)
    p.add_argument('--out', default='outputs/single_roi/single_roi_modules.pth')
    p.add_argument('--resume-adaptive', default='')
    return p.parse_args()


def masked_l1(pred, target, mask):
    err = (pred - target).abs() * mask
    denom = mask.sum() * pred.shape[1] + 1e-6
    return err.sum() / denom


def main():
    args = parse_args()
    cfg, dataset, loader, model, logger = build_env(args, training=True)
    assert args.roi_h > 0 and args.roi_w > 0 and args.age_cap > 0

    bev_channels = int(cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
    center_net = BEVCenterPredictor(bev_channels=bev_channels, age_cap=args.age_cap).cuda()
    pred_net = BEVTemporalPredictor(channels=bev_channels, age_cap=args.age_cap).cuda()

    predictor_loaded = False
    if args.resume_adaptive:
        ck = torch.load(args.resume_adaptive, map_location='cpu')
        if 'center_net' in ck:
            center_net.load_state_dict(ck['center_net'], strict=True)
        if 'pred_net' in ck:
            pred_net.load_state_dict(ck['pred_net'], strict=True)
            predictor_loaded = True

    # Center-only training may optionally use a pretrained predictor to define
    # the oracle refresh utility. Without one, it safely falls back to reuse error.
    if args.stage == 'center' and not predictor_loaded:
        print('[INFO] center stage without a pretrained predictor: oracle target uses reuse error only.')

    params = []
    if args.stage in ['center', 'joint']:
        params += list(center_net.parameters())
    if args.stage in ['predictor', 'joint']:
        params += list(pred_net.parameters())
    assert params
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    amp_enabled = bool(model.use_amp_dict.get('TEST', False))

    for epoch in range(args.epochs):
        center_net.train(args.stage in ['center', 'joint'])
        pred_net.train(args.stage in ['predictor', 'joint'])
        sum_loss = 0.0
        used = 0

        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            load_data_to_gpu(batch)
            token = batch.get('token')
            prev = batch.get('prev')
            prev2 = batch.get('prev2')
            if token is None or prev is None:
                continue
            if prev2 is None:
                prev2 = prev

            with torch.no_grad():
                _, cur = extract_full_bev(model, token)
                _, p1 = extract_full_bev(model, prev)
                _, p2 = extract_full_bev(model, prev2)

            B, C, H, W = cur.shape
            assert C == bev_channels
            age = torch.ones((B, 1, H, W), device=cur.device, dtype=cur.dtype)

            with cuda_autocast(amp_enabled):
                pred = pred_net(p1, p2, age)

                reuse_err = (cur - p1).abs().mean(1, keepdim=True)
                pred_err = (cur - pred.detach()).abs().mean(1, keepdim=True)
                if args.stage == 'center' and predictor_loaded:
                    utility = torch.minimum(reuse_err, pred_err)
                elif args.stage == 'joint' and epoch > 0:
                    # Avoid defining center labels from a random predictor at epoch 0.
                    utility = torch.minimum(reuse_err, pred_err)
                else:
                    utility = reuse_err
                centers, _ = oracle_center_from_error(utility.float(), (args.roi_h, args.roi_w))

                outside = torch.ones((B, 1, H, W), device=cur.device, dtype=cur.dtype)
                for b in range(B):
                    box = center_to_box(centers[b], (args.roi_h, args.roi_w), H, W)
                    outside[b:b+1] -= box_to_mask(box, H, W, cur.device, cur.dtype)
                outside.clamp_(0, 1)

                loss_pred = masked_l1(pred, cur, outside)

                logits = center_net(token['left_img'], token['right_img'], p1, age)
                assert logits.shape == (B, 1, H, W), (logits.shape, cur.shape)
                target = gaussian_heatmap(
                    centers, H, W, sigma=args.sigma,
                    device=logits.device, dtype=logits.dtype,
                )
                target_idx = centers[:, 1] * W + centers[:, 0]
                loss_peak = F.cross_entropy(logits.flatten(1), target_idx.long())
                loss_heat = F.mse_loss(torch.sigmoid(logits), target)
                loss_center = loss_peak + args.lambda_heat * loss_heat

                if args.stage == 'predictor':
                    loss = args.lambda_pred * loss_pred
                elif args.stage == 'center':
                    loss = loss_center
                else:
                    loss = loss_center + args.lambda_pred * loss_pred

            assert torch.isfinite(loss), 'training loss is NaN/Inf'
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optim.step()

            sum_loss += float(loss.item())
            used += 1
            if used <= 3 or used % 20 == 0:
                print(
                    f'epoch={epoch:02d} iter={used:05d} '
                    f'loss={sum_loss/used:.5f} '
                    f'center={float(loss_center.item()):.5f} '
                    f'pred={float(loss_pred.item()):.5f}'
                )

        ckpt = {
            'center_net': center_net.state_dict(),
            'pred_net': pred_net.state_dict(),
            'meta': {
                'roi_h': args.roi_h,
                'roi_w': args.roi_w,
                'sigma': args.sigma,
                'age_cap': args.age_cap,
                'epoch': epoch,
                'base_cfg': args.cfg,
                'base_ckpt': args.ckpt,
            },
        }
        torch.save(ckpt, out)
        print(f'[epoch {epoch}] used={used} mean_loss={sum_loss/max(used,1):.6f} saved={out}')


if __name__ == '__main__':
    main()
