#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


_EXT = None

_CPP_SRC = r"""
#include <torch/extension.h>

void launch_persistent_contention_cuda(
    torch::Tensor sink,
    int64_t blocks,
    int64_t threads,
    double duration_ms
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "launch_persistent_contention",
        &launch_persistent_contention_cuda,
        "Launch persistent arithmetic contention kernel"
    );
}
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void persistent_arithmetic_kernel(
    float* __restrict__ sink,
    unsigned long long duration_cycles
) {
    const unsigned long long t0 = clock64();

    float a0 = 1.00001f + 0.000001f * (float)(threadIdx.x + 1);
    float a1 = 1.00002f + 0.000001f * (float)(blockIdx.x + 1);
    float a2 = 0.99991f;
    float a3 = 1.00013f;
    float a4 = 0.99987f;
    float a5 = 1.00007f;
    float a6 = 0.99993f;
    float a7 = 1.00011f;

    const float b0 = 1.000000119f;
    const float b1 = 0.999999940f;
    const float c0 = 0.000001013f;
    const float c1 = 0.000000977f;

    // Eight independent arithmetic chains give the SM enough independent
    // FP32 FMA work to create smooth compute pressure while keeping memory
    // traffic negligible.
    while ((clock64() - t0) < duration_cycles) {
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            a0 = fmaf(a0, b0, c0);
            a1 = fmaf(a1, b1, c1);
            a2 = fmaf(a2, b0, a0 * 1.0e-7f);
            a3 = fmaf(a3, b1, a1 * 1.0e-7f);
            a4 = fmaf(a4, b0, a2 * 1.0e-7f);
            a5 = fmaf(a5, b1, a3 * 1.0e-7f);
            a6 = fmaf(a6, b0, a4 * 1.0e-7f);
            a7 = fmaf(a7, b1, a5 * 1.0e-7f);
        }
    }

    const float v = (
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
    ) * 1.0e-12f;

    if (threadIdx.x == 0) {
        atomicAdd(sink, v);
    }
}

void launch_persistent_contention_cuda(
    torch::Tensor sink,
    int64_t blocks,
    int64_t threads,
    double duration_ms
) {
    TORCH_CHECK(sink.is_cuda(), "sink must be CUDA tensor");
    TORCH_CHECK(sink.scalar_type() == torch::kFloat32, "sink must be float32");
    TORCH_CHECK(sink.numel() >= 1, "sink must contain at least one element");
    TORCH_CHECK(blocks > 0, "blocks must be > 0");
    TORCH_CHECK(threads > 0 && threads <= 1024, "threads must be in [1,1024]");
    TORCH_CHECK(duration_ms > 0.0, "duration_ms must be > 0");

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // cudaDeviceProp.clockRate is in kHz.  Numerically, one millisecond
    // corresponds to clockRate device-clock cycles.
    const unsigned long long duration_cycles =
        (unsigned long long)(duration_ms * (double)prop.clockRate);

    auto stream = at::cuda::getCurrentCUDAStream(device);

    persistent_arithmetic_kernel<<<
        (unsigned int)blocks,
        (unsigned int)threads,
        0,
        stream.stream()
    >>>(
        sink.data_ptr<float>(),
        duration_cycles
    );

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "persistent_arithmetic_kernel launch failed: ",
        cudaGetErrorString(err)
    );
}
"""


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    # Put the cache in the user's normal torch extension cache unless an
    # explicit directory is requested.
    verbose = os.environ.get("PERSISTENT_CONTENTION_BUILD_VERBOSE", "0") == "1"

    _EXT = load_inline(
        name="streamdsgn_persistent_contention_ext_v1",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=None,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
        ],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=verbose,
    )
    return _EXT


def gpu_sm_count(device: int | None = None) -> int:
    if device is None:
        device = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


class PersistentCudaContention:
    """
    One finite, approximately uniform GPU-compute contention window.

    strength:
        Nominal block/SM ratio.  It is a load-generator parameter, NOT an
        assertion that exactly this fraction of GPU compute is consumed.

        blocks = ceil(strength * number_of_SMs)

    duration_ms:
        Device-side arithmetic kernel duration.

    start_delay_ms:
        Host waits this long after the contention stream has reached the launch
        point before starting the measured model forward.  This makes the model
        start inside the steady part of the contention window.

    Recommended experiment semantics:
        launch -> wait start_delay -> one forward -> wait window end

    This yields a nearly constant contention condition over one forward and
    avoids burst/gap oscillation from repeatedly launching short GEMMs.
    """

    def __init__(
        self,
        strength: float,
        duration_ms: float = 100.0,
        start_delay_ms: float = 5.0,
        threads: int = 256,
        device: int | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        self.device = (
            torch.cuda.current_device()
            if device is None
            else int(device)
        )
        torch.cuda.set_device(self.device)

        self.strength = float(strength)
        self.duration_ms = float(duration_ms)
        self.start_delay_ms = float(start_delay_ms)
        self.threads = int(threads)

        if self.strength <= 0:
            raise ValueError("strength must be > 0")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be > 0")
        if self.start_delay_ms < 0:
            raise ValueError("start_delay_ms must be >= 0")
        if self.start_delay_ms >= self.duration_ms:
            raise ValueError("start_delay_ms must be < duration_ms")

        self.sm_count = gpu_sm_count(self.device)
        self.blocks = max(
            1,
            int(math.ceil(self.strength * self.sm_count)),
        )

        self.stream = torch.cuda.Stream(device=self.device)
        self.sink = torch.zeros(
            (1,),
            device=f"cuda:{self.device}",
            dtype=torch.float32,
        )

        self._launch_event = None
        self._end_event = None

        # Compile/load once at construction time, outside the measurements.
        self.ext = _load_extension()

    def launch(self):
        """
        Launch one finite contention window asynchronously.
        """
        self._launch_event = torch.cuda.Event(enable_timing=False)
        self._end_event = torch.cuda.Event(enable_timing=False)

        with torch.cuda.stream(self.stream):
            self._launch_event.record()
            self.ext.launch_persistent_contention(
                self.sink,
                int(self.blocks),
                int(self.threads),
                float(self.duration_ms),
            )
            self._end_event.record()

        # Ensure the contender stream reached the point immediately before the
        # persistent kernel, then wait a small fixed delay so the kernel is in
        # steady execution before model forward starts.
        self._launch_event.synchronize()
        if self.start_delay_ms > 0:
            time.sleep(self.start_delay_ms / 1000.0)

    def finish(self):
        """
        Wait until this contention window naturally ends.
        Called only AFTER the measured model forward; not part of forward time.
        """
        if self._end_event is not None:
            self._end_event.synchronize()

    @property
    def guaranteed_budget_ms(self) -> float:
        return self.duration_ms - self.start_delay_ms

    def assert_covers_forward(self, forward_ms: float, margin_ms: float = 2.0):
        if float(forward_ms) + float(margin_ms) > self.guaranteed_budget_ms:
            raise RuntimeError(
                "Contention window did not safely cover the whole forward: "
                f"forward={forward_ms:.3f} ms, "
                f"available={self.guaranteed_budget_ms:.3f} ms, "
                f"margin={margin_ms:.3f} ms. "
                "Increase --contention-duration-ms."
            )

    def config(self):
        return {
            "type": "finite_persistent_cuda_arithmetic_kernel",
            "strength": self.strength,
            "sm_count": self.sm_count,
            "blocks": self.blocks,
            "threads": self.threads,
            "duration_ms": self.duration_ms,
            "start_delay_ms": self.start_delay_ms,
            "guaranteed_contention_budget_ms": self.guaranteed_budget_ms,
            "strength_semantics": (
                "nominal blocks/SM ratio; actual contention level is defined "
                "by measured fixed-prefix latency"
            ),
        }


def build_extension_only():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    ext = _load_extension()
    print("[OK] persistent contention CUDA extension loaded:", ext)
    print("[GPU] SM count:", gpu_sm_count())


if __name__ == "__main__":
    build_extension_only()

