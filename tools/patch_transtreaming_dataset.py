from pathlib import Path

path = Path(
    "pcdet/datasets/kitti_streaming/"
    "stereo_kitti_streaming.py"
)

s = path.read_text()

call_old = (
    "        self._maybe_add_mtd_future_tags()\n"
)

call_new = (
    "        self._maybe_add_mtd_future_tags()\n"
    "        self._maybe_add_transtreaming_past_tags()\n"
)

if (
    call_new not in s
    and
    call_old in s
):
    s = s.replace(
        call_old,
        call_new,
        1,
    )

method = r'''
    def _maybe_add_transtreaming_past_tags(self):
        """
        Add arbitrary prevK metadata by following
        the same-scene `prev` chain.

        Used by Transtreaming mixed-speed temporal
        training, e.g. prev3/prev4/prev6/prev8.
        """
        steps = self.dataset_cfg.get(
            'TRANSTREAMING_PAST_STEPS',
            [],
        )

        if steps is None:
            return

        steps = sorted(
            set(
                int(x)
                for x in steps
            )
        )

        if len(steps) == 0:
            return

        if any(
            x < 2
            for x in steps
        ):
            raise ValueError(
                'TRANSTREAMING_PAST_STEPS '
                'must contain integers >= 2: '
                f'{steps}'
            )

        lookup = {}

        for item in self.kitti_infos:
            scene = str(
                item[
                    'sample_idx'
                ][
                    'scene'
                ]
            )

            token = (
                item[
                    'sample_idx'
                ][
                    'frame_tag'
                ].get(
                    'token',
                    '',
                )
            )

            if token == '':
                continue

            key = (
                scene,
                str(token),
            )

            if key in lookup:
                raise RuntimeError(
                    'duplicate Transtreaming '
                    f'token: {key}'
                )

            lookup[key] = item

        valid_count = {
            step: 0
            for step in steps
        }

        invalid_count = {
            step: 0
            for step in steps
        }

        max_step = max(
            steps
        )

        for item in self.kitti_infos:
            scene = str(
                item[
                    'sample_idx'
                ][
                    'scene'
                ]
            )

            frame_tags = (
                item[
                    'sample_idx'
                ][
                    'frame_tag'
                ]
            )

            cur_info = item

            for step in range(
                1,
                max_step + 1,
            ):
                prev_id = (
                    cur_info[
                        'sample_idx'
                    ][
                        'frame_tag'
                    ].get(
                        'prev',
                        '',
                    )
                )

                if (
                    prev_id is None
                    or
                    str(prev_id) == ''
                ):
                    break

                prev_key = (
                    scene,
                    str(prev_id),
                )

                prev_info = lookup.get(
                    prev_key,
                    None,
                )

                if prev_info is None:
                    break

                if step in steps:
                    tag = (
                        f'prev{step}'
                    )

                    frame_tags[
                        tag
                    ] = str(
                        prev_id
                    )

                    item[
                        'infos'
                    ][
                        tag
                    ] = copy.deepcopy(
                        prev_info[
                            'infos'
                        ][
                            'token'
                        ]
                    )

                    valid_count[
                        step
                    ] += 1

                cur_info = prev_info

            for step in steps:
                tag = (
                    f'prev{step}'
                )

                if tag not in frame_tags:
                    frame_tags[
                        tag
                    ] = ''

                    invalid_count[
                        step
                    ] += 1

        if self.logger is not None:
            self.logger.info(
                'Transtreaming past tags: '
                +
                ', '.join(
                    f'prev{step}: '
                    f'valid={valid_count[step]}, '
                    f'invalid={invalid_count[step]}'
                    for step in steps
                )
            )

'''

marker = (
    "    def set_split(self, split):\n"
)

if (
    "def _maybe_add_transtreaming_past_tags"
    not in s
):
    if marker not in s:
        raise RuntimeError(
            "cannot find set_split marker"
        )

    s = s.replace(
        marker,
        method
        +
        marker,
        1,
    )

path.write_text(s)

print(
    "patched:",
    path,
)
