#!/usr/bin/env python3
import argparse
import re
import shutil
import time
from pathlib import Path

MARK = "STREAMDSGN_LATENCY_OPT_V1"


def die(msg):
    raise RuntimeError(msg)


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        die(f"[{label}] expected exactly 1 anchor, found {n}")
    return text.replace(old, new, 1)


def replace_regex_once(text, pattern, repl, label, flags=0):
    out, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        die(f"[{label}] expected exactly 1 regex match, found {n}")
    return out


def method_span(text, method_name, class_name=None):
    start_search = 0
    if class_name is not None:
        m = re.search(rf"^class {re.escape(class_name)}\b.*?:\s*$", text, re.M)
        if not m:
            die(f"class {class_name} not found")
        start_search = m.end()
    m = re.search(rf"^    def {re.escape(method_name)}\s*\(", text[start_search:], re.M)
    if not m:
        die(f"method {method_name} not found")
    start = start_search + m.start()
    nxt = re.search(r"^    def \w+\s*\(", text[start + 1:], re.M)
    end = len(text) if not nxt else start + 1 + nxt.start()
    return start, end


def replace_in_method(text, method_name, old, new, label, class_name=None):
    s, e = method_span(text, method_name, class_name)
    block = text[s:e]
    n = block.count(old)
    if n != 1:
        die(f"[{label}] expected 1 anchor inside {method_name}, found {n}")
    block = block.replace(old, new, 1)
    return text[:s] + block + text[e:]


def backup_file(path, backup_root, repo):
    rel = path.relative_to(repo)
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def patch_submodule(path):
    s = path.read_text()
    if MARK in s:
        print(f"[SKIP] already patched: {path}")
        return False

    old = '''    def forward_stereo_roi(
        self,
        feats,
        base_rect,
        base_halo=4,
    ):'''
    new = '''    def forward_stereo_roi(
        self,
        feats,
        base_rect,
        base_halo=2,
        spp_cache=None,
    ):'''
    s = replace_once(s, old, new, "submodule forward_stereo_roi signature")

    pattern = re.compile(
        r"        # ------------------------------------------------------------\n"
        r"        # Full-context SPP\.\n"
        r".*?"
        r"        x = torch\.cat\(concat_local, dim=1\)\.contiguous\(\)\n",
        re.S,
    )

    replacement = r'''        # ------------------------------------------------------------
        # STREAMDSGN_LATENCY_OPT_V1: ROI + persistent low-resolution SPP cache.
        # ------------------------------------------------------------
        feat_shape = (H, W)
        concat_local = [
            x[..., ey0:ey1, ex0:ex1]
            for x in feats[self.start_level:]
        ]

        if self.with_spp:
            if spp_cache is None or len(spp_cache) != len(self.spp_branches):
                raise RuntimeError(
                    'ROI SPP cache is missing/uninitialized. '
                    'Each scene must start from a Full A frame.'
                )

            yy = torch.arange(
                ey0, ey1, device=feats[-1].device, dtype=torch.float32
            )
            xx = torch.arange(
                ex0, ex1, device=feats[-1].device, dtype=torch.float32
            )
            gy = torch.zeros_like(yy) if H <= 1 else (2.0 * yy / (H - 1) - 1.0)
            gx = torch.zeros_like(xx) if W <= 1 else (2.0 * xx / (W - 1) - 1.0)
            gy, gx = torch.meshgrid(gy, gx, indexing='ij')
            spp_grid = torch.stack([gx, gy], dim=-1)[None]

            for branch_idx, branch_module in enumerate(self.spp_branches):
                cache = spp_cache[branch_idx]
                if cache is None:
                    raise RuntimeError(
                        f'ROI SPP cache branch {branch_idx} is uninitialized'
                    )

                pool = branch_module[0]
                if not isinstance(pool, nn.AvgPool2d):
                    raise RuntimeError(
                        'ROI SPP cache currently supports fixed AvgPool2d only; '
                        f'got {type(pool).__name__}'
                    )

                kh, kw = (
                    pool.kernel_size if isinstance(pool.kernel_size, tuple)
                    else (pool.kernel_size, pool.kernel_size)
                )
                sh, sw = (
                    pool.stride if isinstance(pool.stride, tuple)
                    else (pool.stride, pool.stride)
                )
                ph, pw = (
                    pool.padding if isinstance(pool.padding, tuple)
                    else (pool.padding, pool.padding)
                )
                if (ph, pw) != (0, 0) or pool.ceil_mode:
                    raise RuntimeError(
                        'ROI SPP cache expects padding=0, ceil_mode=False'
                    )

                Hs, Ws = cache.shape[-2:]
                py0 = max(0, y0 // sh)
                px0 = max(0, x0 // sw)
                py1 = min(Hs, (y1 + sh - 1) // sh)
                px1 = min(Ws, (x1 + sw - 1) // sw)

                if py1 > py0 and px1 > px0:
                    iy0 = py0 * sh
                    ix0 = px0 * sw
                    iy1 = (py1 - 1) * sh + kh
                    ix1 = (px1 - 1) * sw + kw

                    local = feats[-1][..., iy0:iy1, ix0:ix1].contiguous()
                    local = pool(local)
                    for sub in list(branch_module.children())[1:]:
                        local = sub(local)

                    expected = (py1 - py0, px1 - px0)
                    if tuple(local.shape[-2:]) != expected:
                        raise RuntimeError(
                            f'ROI SPP local shape mismatch: '
                            f'got={tuple(local.shape[-2:])}, expected={expected}'
                        )
                    cache[..., py0:py1, px0:px1].copy_(local)

                spp_local = F.grid_sample(
                    cache,
                    spp_grid.to(dtype=cache.dtype),
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True,
                )
                concat_local.append(spp_local)

        x = torch.cat(concat_local, dim=1).contiguous()
'''

    s2, n = pattern.subn(replacement, s, count=1)
    if n != 1:
        die(f"[submodule SPP block] expected 1 match, found {n}")
    s = s2
    s = s.replace(
        "from torchvision.ops import DeformConv2d\n",
        "from torchvision.ops import DeformConv2d\n\n# " + MARK + "\n",
        1,
    )
    path.write_text(s)
    print(f"[PATCH] {path}")
    return True


def patch_backbone(path):
    s = path.read_text()
    if MARK in s:
        print(f"[SKIP] already patched: {path}")
        return False

    s = replace_regex_once(
        s,
        r"self\._adaptive_a_halos\s*=\s*\{.*?\}",
        "self._adaptive_a_halos = {1: 2, 2: 2, 3: 2}",
        "A halos",
        flags=re.S,
    )
    s = replace_regex_once(
        s,
        r"self\._adaptive_a_neck_halo\s*=\s*\d+",
        "self._adaptive_a_neck_halo = 2",
        "A neck halo",
    )

    anchor = "        self.feature_neck = feature_extraction_neck(model_cfg.feature_neck)\n"
    insert = anchor + '''        # STREAMDSGN_LATENCY_OPT_V1: side-specific low-res SPP caches.
        self._adaptive_spp_capture_side = None
        self._adaptive_spp_cache = {'left': None, 'right': None}
        self._adaptive_spp_hook_handles = []
        if getattr(self.feature_neck, 'with_spp', False):
            num_spp = len(self.feature_neck.spp_branches)
            self._adaptive_spp_cache = {
                'left': [None] * num_spp,
                'right': [None] * num_spp,
            }
            for _spp_idx, _spp_branch in enumerate(self.feature_neck.spp_branches):
                def _capture_spp(module, inputs, output, idx=_spp_idx):
                    side = self._adaptive_spp_capture_side
                    if side in ('left', 'right'):
                        self._adaptive_spp_cache[side][idx] = output.detach()
                self._adaptive_spp_hook_handles.append(
                    _spp_branch.register_forward_hook(_capture_spp)
                )
'''
    s = replace_once(s, anchor, insert, "SPP hook init")

    anchor = '''        self.prepare_depth(self.point_cloud_range, in_camera_view=False)
        self.prepare_coordinates_3d(self.point_cloud_range, voxel_size, grid_size)
'''
    insert = anchor + '''        # STREAMDSGN_LATENCY_OPT_V1: resident static tensors.
        for _name in (
            'downsampled_depth', 'downsampledx2_depth', 'depth', 'coordinates_3d'
        ):
            _tensor = getattr(self, _name)
            delattr(self, _name)
            self.register_buffer(_name, _tensor.contiguous(), persistent=False)
            self.register_buffer(
                f'_adaptive_{_name}_half',
                _tensor.half().contiguous(),
                persistent=False,
            )
'''
    s = replace_once(s, anchor, insert, "register static buffers")

    s = replace_once(
        s,
        "            left_stereo_feat, left_sem_feat = self.feature_neck(left_features)",
        '''            self._adaptive_spp_capture_side = 'left'
            try:
                left_stereo_feat, left_sem_feat = self.feature_neck(left_features)
            finally:
                self._adaptive_spp_capture_side = None''',
        "full left SPP capture",
    )

    s = replace_once(
        s,
        '''                right_stereo_feat, right_sem_feat = \\
                    self.feature_neck(right_features)''',
        '''                self._adaptive_spp_capture_side = 'right'
                try:
                    right_stereo_feat, right_sem_feat = self.feature_neck(right_features)
                finally:
                    self._adaptive_spp_capture_side = None''',
        "full right SPP capture",
    )

    s = replace_once(
        s,
        '''        return_stage_features=False,
    ):''',
        '''        return_stage_features=False,
        side=None,
    ):''',
        "deep full helper signature",
    )

    s = replace_in_method(
        s,
        "forward_2d_deep_full_from_shallow",
        "        stereo_feature, sem_feature = self.feature_neck([img] + feats)",
        '''        old_side = self._adaptive_spp_capture_side
        self._adaptive_spp_capture_side = side
        try:
            stereo_feature, sem_feature = self.feature_neck([img] + feats)
        finally:
            self._adaptive_spp_capture_side = old_side''',
        "deep full helper SPP capture",
    )

    ms, me = method_span(s, "forward_2d_adaptive")
    block = s[ms:me]
    old_call = '''                    return_stage_features=True,
                )'''
    cnt = block.count(old_call)
    if cnt != 2:
        die(f"[forward_2d_adaptive side propagation] expected 2 calls, found {cnt}")
    block = block.replace(
        old_call,
        '''                    return_stage_features=True,
                    side=side,
                )''',
    )
    s = s[:ms] + block + s[me:]

    s = replace_once(
        s,
        '''            self.feature_neck.forward_stereo_roi(
                feats,
                base_rect=rect,
                base_halo=self._adaptive_a_neck_halo,
            )''',
        '''            self.feature_neck.forward_stereo_roi(
                feats,
                base_rect=rect,
                base_halo=self._adaptive_a_neck_halo,
                spp_cache=self._adaptive_spp_cache[side],
            )''',
        "backbone adaptive SPP cache use",
    )

    s = replace_once(
        s,
        "            downsampled_depth = self.downsampled_depth.cuda().half() if self.use_amp else self.downsampled_depth.cuda()",
        '''            downsampled_depth = (
                self._adaptive_downsampled_depth_half
                if self.use_amp else self.downsampled_depth
            )''',
        "resident downsampled depth",
    )

    s = replace_once(
        s,
        "        coordinates_3d = self.coordinates_3d.cuda().half() if self.use_amp else self.coordinates_3d.cuda()",
        '''        coordinates_3d = (
            self._adaptive_coordinates_3d_half
            if self.use_amp else self.coordinates_3d
        )''',
        "resident coordinates_3d",
    )

    s = s.replace("import numpy as np\n", "import numpy as np\n\n# " + MARK + "\n", 1)
    path.write_text(s)
    print(f"[PATCH] {path}")
    return True


def patch_cost_volume(path):
    s = path.read_text()
    if MARK in s:
        print(f"[SKIP] already patched: {path}")
        return False
    start, end = method_span(s, "forward_roi", class_name="BuildCostVolume")
    block = s[start:end]
    casts = '''        if left.dtype == torch.float16:
            left = left.float()

        if right.dtype == torch.float16:
            right = right.float()

        if shift.dtype == torch.float16:
            shift = shift.float()

'''
    if block.count(casts) != 1:
        die(f"[cost_volume forward_roi casts] expected 1 block, found {block.count(casts)}")
    block = block.replace(
        casts,
        '''        # STREAMDSGN_LATENCY_OPT_V1: ROI CUDA accepts FP16 directly.
        if left.dtype != right.dtype or left.dtype != shift.dtype:
            raise RuntimeError(
                f'DPS ROI dtype mismatch: left={left.dtype}, right={right.dtype}, shift={shift.dtype}'
            )

''',
        1,
    )
    s = s[:start] + block + s[end:]
    s = s.replace("from torch import nn\n", "from torch import nn\n\n# " + MARK + "\n", 1)
    path.write_text(s)
    print(f"[PATCH] {path}")
    return True


def patch_cuda(path):
    s = path.read_text()
    if MARK in s:
        print(f"[SKIP] already patched: {path}")
        return False
    old = '''  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "BuildDpsCostVolume_forward_roi",'''
    new = '''  // STREAMDSGN_LATENCY_OPT_V1: ROI inference accepts FP16 directly.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      left.scalar_type(),
      "BuildDpsCostVolume_forward_roi",'''
    s = replace_once(s, old, new, "CUDA ROI half dispatch")
    anchor = '''  AT_ASSERTM(
      left.size(0) == shift.size(0),
      "Image and shift batch must match");
'''
    insert = anchor + '''
  AT_ASSERTM(
      left.scalar_type() == right.scalar_type()
      && left.scalar_type() == shift.scalar_type(),
      "left/right/shift dtypes must match for DPS ROI");
'''
    s = replace_once(s, anchor, insert, "CUDA dtype assert")
    s = "// " + MARK + "\n" + s
    path.write_text(s)
    print(f"[PATCH] {path}")
    return True


def patch_evaluator(path):
    s = path.read_text()
    if MARK in s:
        print(f"[SKIP] already patched: {path}")
        return False

    s = s.replace('p.add_argument("--b_halo", type=int, default=1)', 'p.add_argument("--b_halo", type=int, default=2)')
    s = s.replace('p.add_argument("--cd_halo", type=int, default=12)', 'p.add_argument("--cd_halo", type=int, default=2)')

    m = re.search(r"^def stage_plan_gpu\(q, levels, align\):\n", s, re.M)
    if not m:
        die("[evaluator integral planner] stage_plan_gpu not found")
    nxt = re.search(r"^def \w+\(", s[m.end():], re.M)
    if not nxt:
        die("[evaluator integral planner] next top-level def not found")
    end = m.end() + nxt.start()

    new_block = '''def stage_plan_gpu(q, levels, align):
    # STREAMDSGN_LATENCY_OPT_V1: one integral image per Q map.
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 1:
        raise RuntimeError(f"Expected Q [1,1,H,W], got {tuple(q.shape)}")
    H, W = q.shape[-2:]
    qf = q.float()
    integ = F.pad(qf, (1, 0, 1, 0), mode="constant", value=0.0)
    integ = integ.cumsum(dim=-2).cumsum(dim=-1)
    plans = []
    for level, ratio in enumerate(levels):
        h, w = roi_hw_from_ratio(H, W, ratio, align=align)
        score = (
            integ[..., h:, w:] - integ[..., :-h, w:]
            - integ[..., h:, :-w] + integ[..., :-h, :-w]
        )
        flat = score.flatten(1)
        value, idx = flat.max(dim=1)
        plans.append({
            "level": level, "ratio": float(ratio), "h": int(h), "w": int(w),
            "Wv": int(score.shape[-1]), "value_gpu": value[0], "idx_gpu": idx[0],
        })
    return plans


'''
    s = s[:m.start()] + new_block + s[end:]

    class_m = re.search(r"^class DeadlineScheduler:\s*$", s, re.M)
    if not class_m:
        die("DeadlineScheduler not found")
    choose_m = re.search(r"^    def choose\(self, plans, deadline_ms\):\s*$", s[class_m.end():], re.M)
    if not choose_m:
        die("DeadlineScheduler.choose not found")
    pos = class_m.end() + choose_m.start()

    choose_gpu = '''    def choose_gpu(self, gpu_plans, deadline_ms):
        # STREAMDSGN_LATENCY_OPT_V1: GPU 5x5x5 scheduler.
        device = gpu_plans["A"][0]["value_gpu"].device
        K = len(self.levels)
        if not hasattr(self, "_latency_lut_cpu"):
            vals = []
            for a in range(K):
                for b in range(K):
                    for cd in range(K):
                        vals.append(float(self.actions[f"{a},{b},{cd}"]["safe_ms"]) + self.extra_ms)
            self._latency_lut_cpu = torch.tensor(vals, dtype=torch.float32).view(K, K, K)
            self._latency_lut_gpu = None
            self._latency_lut_device = None
        if self._latency_lut_gpu is None or self._latency_lut_device != device:
            self._latency_lut_gpu = self._latency_lut_cpu.to(device=device)
            self._latency_lut_device = device

        ua = torch.stack([x["value_gpu"].float() for x in gpu_plans["A"]])
        ub = torch.stack([x["value_gpu"].float() for x in gpu_plans["B"]])
        uc = torch.stack([x["value_gpu"].float() for x in gpu_plans["CD"]])
        util = self.lambda_a * ua[:, None, None] + self.lambda_b * ub[None, :, None] + self.lambda_cd * uc[None, None, :]
        lat = self._latency_lut_gpu
        feasible = lat <= float(deadline_ms)
        feasible_count = feasible.sum()
        no_feasible = feasible_count == 0
        masked = util.masked_fill(~feasible, -float("inf"))
        max_utility = masked.max()
        # Preserve the old scheduler's latency tie-break among actions with
        # the same maximum utility.
        best_utility_mask = feasible & (util >= max_utility - 1e-12)
        tie_lat = lat.masked_fill(~best_utility_mask, float("inf"))
        best_flat = tie_lat.reshape(-1).argmin()
        fastest_flat = lat.reshape(-1).argmin()
        best_flat = torch.where(no_feasible, fastest_flat, best_flat)

        a_gpu = best_flat // (K * K)
        rem = best_flat % (K * K)
        b_gpu = rem // K
        cd_gpu = rem % K
        idx_a = torch.stack([x["idx_gpu"] for x in gpu_plans["A"]])[a_gpu]
        idx_b = torch.stack([x["idx_gpu"] for x in gpu_plans["B"]])[b_gpu]
        idx_c = torch.stack([x["idx_gpu"] for x in gpu_plans["CD"]])[cd_gpu]

        selected_lat = lat[a_gpu, b_gpu, cd_gpu]
        selected_util = util[a_gpu, b_gpu, cd_gpu]
        packed = torch.stack([
            a_gpu.float(), b_gpu.float(), cd_gpu.float(),
            idx_a.float(), idx_b.float(), idx_c.float(),
            feasible_count.float(), no_feasible.float(),
            selected_lat.float(), selected_util.float(),
        ]).detach().cpu().tolist()
        a, b, cd, ia, ib, ic, nfeas, nofeas = [int(x) for x in packed[:8]]
        selected_lat_cpu = float(packed[8])
        selected_util_cpu = float(packed[9])

        def _plan(stage, level, idx):
            rec = gpu_plans[stage][level]
            y0 = idx // rec["Wv"]
            x0 = idx % rec["Wv"]
            return {
                "level": level, "ratio": rec["ratio"], "h": rec["h"], "w": rec["w"],
                "rect": (y0, y0 + rec["h"], x0, x0 + rec["w"]),
            }

        return {
            "key": f"{a},{b},{cd}", "a": a, "b": b, "cd": cd,
            "estimated_safe_ms": selected_lat_cpu,
            "utility": selected_util_cpu,
            "feasible_count": nfeas, "no_feasible_action": bool(nofeas),
        }, {"A": _plan("A", a, ia), "B": _plan("B", b, ib), "CD": _plan("CD", cd, ic)}

'''
    s = s[:pos] + choose_gpu + s[pos:]

    old = '''        plans = finalize_stage_plans(gpu_plans)
        choice = self.scheduler.choose(plans, deadline_ms)

        a = choice["a"]
        b = choice["b"]
        cd = choice["cd"]
        a_rect = plans["A"][a]["rect"]
        b_rect = plans["B"][b]["rect"]
        cd_rect = plans["CD"][cd]["rect"]'''
    new = '''        choice, selected_plans = self.scheduler.choose_gpu(gpu_plans, deadline_ms)

        a = choice["a"]
        b = choice["b"]
        cd = choice["cd"]
        a_rect = selected_plans["A"]["rect"]
        b_rect = selected_plans["B"]["rect"]
        cd_rect = selected_plans["CD"]["rect"]'''
    s = replace_once(s, old, new, "GPU scheduler call")

    s = replace_once(
        s,
        '''            stereo_patch, output_rect = bb.feature_neck.forward_stereo_roi(
                feats,
                base_rect=rect,
                base_halo=bb._adaptive_a_neck_halo,
            )''',
        '''            stereo_patch, output_rect = bb.feature_neck.forward_stereo_roi(
                feats,
                base_rect=rect,
                base_halo=bb._adaptive_a_neck_halo,
                spp_cache=bb._adaptive_spp_cache[side],
            )''',
        "evaluator SPP cache",
    )

    s = replace_once(
        s,
        '''            downsampled_depth = (
                bb.downsampled_depth.cuda().half()
                if bb.use_amp
                else bb.downsampled_depth.cuda()
            )''',
        '''            downsampled_depth = (
                bb._adaptive_downsampled_depth_half if bb.use_amp else bb.downsampled_depth
            )''',
        "evaluator downsampled buffer",
    )

    s = replace_once(
        s,
        '''        coordinates_3d = bb.coordinates_3d.to(device="cuda")
        if bb.use_amp:
            coordinates_3d = coordinates_3d.half()''',
        '''        coordinates_3d = (
            bb._adaptive_coordinates_3d_half if bb.use_amp else bb.coordinates_3d
        )''',
        "evaluator coordinates buffer",
    )

    s = s.replace(
        '''    coordinates_3d = (
        bb.coordinates_3d.cuda().half()
        if bb.use_amp
        else bb.coordinates_3d.cuda()
    )''',
        '''    coordinates_3d = (
        bb._adaptive_coordinates_3d_half if bb.use_amp else bb.coordinates_3d
    )''',
    )

    s = s.replace("import argparse\n", "import argparse\n\n# " + MARK + "\n", 1)
    path.write_text(s)
    print(f"[PATCH] {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    rels = [
        Path("pcdet/models/backbones_3d_stream/submodule.py"),
        Path("pcdet/models/backbones_3d_stream/stream_dsgn2_backbone.py"),
        Path("pcdet/models/backbones_3d_stream/cost_volume.py"),
        Path("pcdet/ops/build_dps_cost_volume/src/BuildDpsCostVolume_cuda.cu"),
        Path("tools/eval_adaptive_true_stream.py"),
    ]
    originals = [repo / r for r in rels]
    for p in originals:
        if not p.exists():
            die(f"missing file: {p}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = repo / f"latency_opt_backup_{stamp}"
    if not args.no_backup:
        for p in originals:
            backup_file(p, backup_root, repo)
        print(f"[BACKUP] {backup_root}")

    # Transactional patching: all edits and Python syntax checks happen on a
    # temporary copy first. The repository is replaced only after every anchor
    # and every py_compile check succeeds.
    work_root = repo / f".latency_opt_work_{stamp}"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_files = []
    for src, rel in zip(originals, rels):
        dst = work_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        work_files.append(dst)

    try:
        patch_submodule(work_files[0])
        patch_backbone(work_files[1])
        patch_cost_volume(work_files[2])
        patch_cuda(work_files[3])
        patch_evaluator(work_files[4])

        import py_compile
        for wf, rel in zip(work_files, rels):
            if rel.suffix == ".py":
                py_compile.compile(str(wf), doraise=True)
                print(f"[PYCOMPILE PASS] {rel}")

        for wf, dst in zip(work_files, originals):
            shutil.copy2(wf, dst)
            print(f"[COMMIT] {dst.relative_to(repo)}")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    print("\nPATCH COMPLETE")
    print("Next: python setup.py build_ext --inplace")


if __name__ == "__main__":
    main()