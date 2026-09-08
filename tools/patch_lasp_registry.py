#!/usr/bin/env python3

from pathlib import Path
import re


p = Path(
    'pcdet/models/detectors_stream/__init__.py'
)

if not p.is_file():
    raise FileNotFoundError(
        p
    )

s = p.read_text()

import_line = (
    'from .lasp_stream import LASP_STREAM\n'
)

if (
    import_line.strip()
    not in s
):
    s = (
        import_line
        + s
    )

if (
    "'stream_lasp'"
    not in s
    and
    '"stream_lasp"'
    not in s
):

    m = re.search(
        r'__all__\s*=\s*\{'
        r'(?P<body>.*?)'
        r'\n\}',
        s,
        flags=re.S,
    )

    if not m:
        raise RuntimeError(
            'could not locate '
            'detectors_stream.__all__ '
            'dictionary; inspect file '
            'before patching'
        )

    body = m.group(
        'body'
    )

    insertion = (
        body.rstrip()
        +
        "\n    "
        "'stream_lasp': LASP_STREAM,"
    )

    s = (
        s[
            :m.start(
                'body'
            )
        ]
        + insertion
        + s[
            m.end(
                'body'
            ):
        ]
    )

p.write_text(
    s
)

print(
    f'patched {p}'
)

for line in s.splitlines():
    if (
        'lasp'
        in line.lower()
    ):
        print(
            line
        )
