#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly 1 anchor, found {n}")
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='.', help='streamDSGN repository root')
    args = p.parse_args()

    bundle = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve()
    target = repo / 'pcdet/models/backbones_3d_stream/stream_dsgn2_backbone.py'
    if not target.is_file():
        raise FileNotFoundError(f'Not a streamDSGN repo root: missing {target}')

    text = target.read_text()

    # Hunk 1: training state/mix helpers.
    marker1 = 'Differentiable adaptive-training surrogate'
    if marker1 not in text:
        old = """    def clear_adaptive_a_profile(self):\n        self._ensure_adaptive_a_runtime()\n        self._adaptive_a_enabled = False\n        self._adaptive_a_ratio = 1.0\n        self._adaptive_a_position_mode = 'center'\n"""
        new = old + """

    # ================================================================
    # Differentiable adaptive-training surrogate
    #
    # This is deliberately separate from the physical ROI inference path.
    # Training computes the full current stage, then composes current/cache
    # with a straight-through rectangular mask. This keeps ROI location
    # differentiable while preserving hard rectangular execution semantics in
    # the forward pass. Physical ROI kernels are profiled/used after training.
    # ================================================================

    def set_adaptive_training_state(
        self,
        a_mask,
        b_mask,
        a_cache_left,
        a_cache_right,
        b_cache,
    ):
        self._adaptive_training_state = {
            'a_mask': a_mask,
            'b_mask': b_mask,
            'a_cache_left': a_cache_left,
            'a_cache_right': a_cache_right,
            'b_cache': b_cache,
        }

    def clear_adaptive_training_state(self):
        self._adaptive_training_state = None

    @staticmethod
    def _adaptive_training_mix_2d(current, cache, mask):
        if cache is None:
            return current
        if tuple(cache.shape) != tuple(current.shape):
            raise RuntimeError(
                f'adaptive A cache shape mismatch: cache={tuple(cache.shape)}, '
                f'current={tuple(current.shape)}'
            )
        m = F.interpolate(mask.float(), size=current.shape[-2:], mode='nearest')
        m = m.to(device=current.device, dtype=current.dtype)
        return m * current + (1.0 - m) * cache.to(current.dtype)

    @staticmethod
    def _adaptive_training_mix_3d(current, cache, mask):
        if cache is None:
            return current
        if tuple(cache.shape) != tuple(current.shape):
            raise RuntimeError(
                f'adaptive B cache shape mismatch: cache={tuple(cache.shape)}, '
                f'current={tuple(current.shape)}'
            )
        m = F.interpolate(mask.float(), size=current.shape[-2:], mode='nearest')
        m = m[:, :, None].to(device=current.device, dtype=current.dtype)
        return m * current + (1.0 - m) * cache.to(current.dtype)
"""
        text = replace_once(text, old, new, 'insert training helpers')

    # Hunk 2: A-stage differentiable composition.
    marker2 = 'Training-time A-stage interface composition.'
    if marker2 not in text:
        old = """        # Expose A_s output for the future importance predictor.\n        batch_dict['left_shallow_feature'] = left_shallow\n        if right_shallow is not None:\n            batch_dict['right_shallow_feature'] = right_shallow\n"""
        new = """        # ------------------------------------------------------------
        # Training-time A-stage interface composition.
        # Q_A lives on the 80x312 A_d native grid; the mask is resized to the
        # dense stereo-output grid (320x1248 in the current configuration).
        # ------------------------------------------------------------
        train_state = getattr(self, '_adaptive_training_state', None)
        batch_dict['adaptive_stage_a_left_current'] = left_stereo_feat
        if not self.mono:
            batch_dict['adaptive_stage_a_right_current'] = right_stereo_feat

        if train_state is not None:
            if self.cat_img_feature or self.cat_right_img_feature:
                raise NotImplementedError(
                    'The current adaptive-training surrogate assumes '
                    'cat_img_feature=False and cat_right_img_feature=False, '
                    'which matches the supplied StreamDSGN config.'
                )
            left_stereo_feat = self._adaptive_training_mix_2d(
                left_stereo_feat,
                train_state['a_cache_left'],
                train_state['a_mask'],
            )
            if not self.mono:
                right_stereo_feat = self._adaptive_training_mix_2d(
                    right_stereo_feat,
                    train_state['a_cache_right'],
                    train_state['a_mask'],
                )

        batch_dict['adaptive_stage_a_left'] = left_stereo_feat
        if not self.mono:
            batch_dict['adaptive_stage_a_right'] = right_stereo_feat

        # Expose A_s output for the importance predictor.
        batch_dict['left_shallow_feature'] = left_shallow
        if right_shallow is not None:
            batch_dict['right_shallow_feature'] = right_shallow
"""
        text = replace_once(text, old, new, 'insert A-stage composition')

    # Hunk 3: B-stage differentiable composition.
    marker3 = 'Training-time B-stage interface composition.'
    if marker3 not in text:
        old = """            if self.use_stereo_out_type == \"feature\":\n                out = all_costs[-1]\n            elif self.use_stereo_out_type == \"prob\":\n                out = cost_softmax_i.unsqueeze(1)\n            elif self.use_stereo_out_type == \"cost\":\n                out = upcost_i.unsqueeze(1)\n            else:\n                raise ValueError('wrong self.use_stereo_out_type option')\n"""
        new = old + """

            # --------------------------------------------------------
            # Training-time B-stage interface composition.
            # B native ROI grid is H=80,W=312 in the current configuration.
            # The mask is broadcast over disparity/depth D.
            # --------------------------------------------------------
            batch_dict['adaptive_stage_b_current'] = out
            train_state = getattr(self, '_adaptive_training_state', None)
            if train_state is not None:
                out = self._adaptive_training_mix_3d(
                    out,
                    train_state['b_cache'],
                    train_state['b_mask'],
                )
            batch_dict['adaptive_stage_b'] = out
"""
        text = replace_once(text, old, new, 'insert B-stage composition')

    backup = target.with_suffix(target.suffix + '.before_adaptive_training')
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(text)

    copies = [
        ('pcdet/models/adaptive_single_roi/stage_importance_predictor.py',
         'pcdet/models/adaptive_single_roi/stage_importance_predictor.py'),
        ('pcdet/models/adaptive_single_roi/differentiable_roi.py',
         'pcdet/models/adaptive_single_roi/differentiable_roi.py'),
        ('pcdet/models/adaptive_single_roi/__init__.py',
         'pcdet/models/adaptive_single_roi/__init__.py'),
        ('tools/train_adaptive_joint.py', 'tools/train_adaptive_joint.py'),
    ]
    for src_rel, dst_rel in copies:
        src = bundle / src_rel
        dst = repo / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f'copied: {dst_rel}')

    print(f'modified: {target.relative_to(repo)}')
    print(f'backup:   {backup.relative_to(repo)}')
    print('DONE')


if __name__ == '__main__':
    main()
