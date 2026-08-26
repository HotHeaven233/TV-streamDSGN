#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


DEFAULT_CFG = 'configs/stream/kitti_models/stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml'
DEFAULT_CKPT = 'extra_data/checkpoint_epoch_20.pth'


def add_base_args(parser: argparse.ArgumentParser, training=False):
    parser.add_argument('--cfg', default=DEFAULT_CFG)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--max-batches', type=int, default=0,
                        help='0 means no truncation. For semantic smoke tests, official KITTI AP is skipped when truncated.')
    if training:
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--lr', type=float, default=2e-4)
    return parser


def cuda_autocast(enabled: bool):
    """Use the current AMP API while keeping the call site concise."""
    return torch.amp.autocast(device_type='cuda', enabled=bool(enabled))


def build_env(args, training=False):
    # Each experiment is run as a separate Python process, so using the global
    # OpenPCDet cfg object here is safe and matches the repository's own tools.
    cfg_from_yaml_file(args.cfg, cfg)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = common_utils.create_logger(rank=0)
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=training,
    )
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return cfg, dataset, loader, model, logger


def extract_full_bev(model, frame_dict):
    """Full teacher feature. Never use this function for latency claims."""
    x = dict(frame_dict)
    enabled = bool(model.use_amp_dict.get('TEST', False))
    with torch.no_grad(), cuda_autocast(enabled):
        for m in model.feature_extractor:
            x = m(x)
    assert 'spatial_features' in x, 'feature_extractor did not produce spatial_features'
    return x, x['spatial_features']


def find_module(model, class_name):
    for m in model.module_list:
        if type(m).__name__ == class_name:
            return m
    raise KeyError(class_name)


def percentile(xs, q):
    if not xs:
        return float('nan')
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=_json_default
        )
    )


def scalar_string(v):
    """Convert common collate outputs (str/list/ndarray) to one stable string."""
    if v is None:
        return ''
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return ''
        return str(v.reshape(-1)[0])
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return ''
        return str(v[0])
    return str(v)
