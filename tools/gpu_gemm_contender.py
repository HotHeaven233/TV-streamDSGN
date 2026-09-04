#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import os
import time
from pathlib import Path

import torch

RUN = True


def _stop(*_):
    global RUN
    RUN = False


def parse_args():
    p = argparse.ArgumentParser(
        description='Fixed GEMM background GPU contender.'
    )
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--matrix-size', type=int, default=4096)
    p.add_argument('--busy-repeats', type=int, required=True)
    p.add_argument('--idle-ms', type=float, default=2.0)
    p.add_argument('--dtype', choices=('fp16', 'bf16', 'fp32'), default='fp16')
    p.add_argument('--warmup-repeats', type=int, default=8)
    p.add_argument('--ready-file', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    if args.busy_repeats < 1:
        raise ValueError('--busy-repeats must be >= 1')

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }[args.dtype]

    n = int(args.matrix_size)
    device = torch.device(f'cuda:{args.device}')

    # Fixed workload shape for all contention levels.  Only busy_repeats varies.
    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(int(args.warmup_repeats)):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()

    ready = Path(args.ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(
        f'pid={os.getpid() if False else "ready"}\n'
        f'matrix_size={n}\n'
        f'busy_repeats={args.busy_repeats}\n'
        f'idle_ms={args.idle_ms}\n'
        f'dtype={args.dtype}\n'
    )

    print(
        f'[READY] GEMM contender: N={n}, repeats={args.busy_repeats}, '
        f'idle_ms={args.idle_ms}, dtype={args.dtype}',
        flush=True,
    )

    # One level = one fixed repeating background compute pattern.
    # The operation and matrix size remain identical across levels; only the
    # number of GEMMs per burst changes.
    with torch.no_grad():
        while RUN:
            for _ in range(int(args.busy_repeats)):
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
            if args.idle_ms > 0:
                time.sleep(args.idle_ms / 1000.0)

    torch.cuda.synchronize()
    print('[STOP] GEMM contender exited', flush=True)


if __name__ == '__main__':
    main()

