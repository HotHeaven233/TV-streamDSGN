#!/usr/bin/env python3
from __future__ import annotations
import math, os, time
import torch
from torch.utils.cpp_extension import load_inline
_EXT=None
_CPP_SRC=r"""
#include <torch/extension.h>
void launch_smooth_contention_chain_cuda(torch::Tensor sink,int64_t blocks,int64_t threads,double slice_ms,int64_t num_slices);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("launch_smooth_contention_chain",&launch_smooth_contention_chain_cuda,"Launch short-kernel compute-contention chain");}
"""
_CUDA_SRC=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void smooth_arithmetic_slice(float* __restrict__ sink,unsigned long long duration_cycles){
    const unsigned long long t0=clock64();
    float a0=1.00001f+0.000001f*(float)(threadIdx.x+1),a1=1.00002f+0.000001f*(float)(blockIdx.x+1);
    float a2=0.99991f,a3=1.00013f,a4=0.99987f,a5=1.00007f,a6=0.99993f,a7=1.00011f;
    const float b0=1.000000119f,b1=0.999999940f,c0=0.000001013f,c1=0.000000977f;
    while((clock64()-t0)<duration_cycles){
        #pragma unroll 8
        for(int i=0;i<8;++i){
            a0=fmaf(a0,b0,c0); a1=fmaf(a1,b1,c1); a2=fmaf(a2,b0,a0*1e-7f); a3=fmaf(a3,b1,a1*1e-7f);
            a4=fmaf(a4,b0,a2*1e-7f); a5=fmaf(a5,b1,a3*1e-7f); a6=fmaf(a6,b0,a4*1e-7f); a7=fmaf(a7,b1,a5*1e-7f);
        }
    }
    if(threadIdx.x==0) atomicAdd(sink,(a0+a1+a2+a3+a4+a5+a6+a7)*1e-12f);
}
void launch_smooth_contention_chain_cuda(torch::Tensor sink,int64_t blocks,int64_t threads,double slice_ms,int64_t num_slices){
    TORCH_CHECK(sink.is_cuda(),"sink must be CUDA tensor"); TORCH_CHECK(sink.scalar_type()==torch::kFloat32,"sink must be float32");
    TORCH_CHECK(blocks>0,"blocks must be >0"); TORCH_CHECK(threads>0&&threads<=1024,"threads invalid");
    TORCH_CHECK(slice_ms>0.0,"slice_ms must be >0"); TORCH_CHECK(num_slices>0,"num_slices must be >0");
    int device=0; cudaGetDevice(&device); cudaDeviceProp prop; cudaGetDeviceProperties(&prop,device);
    const unsigned long long cycles=(unsigned long long)(slice_ms*(double)prop.clockRate);
    auto stream=at::cuda::getCurrentCUDAStream(device);
    for(int64_t i=0;i<num_slices;++i){smooth_arithmetic_slice<<<(unsigned int)blocks,(unsigned int)threads,0,stream.stream()>>>(sink.data_ptr<float>(),cycles);}
    const cudaError_t err=cudaGetLastError(); TORCH_CHECK(err==cudaSuccess,"smooth_arithmetic_slice launch failed: ",cudaGetErrorString(err));
}
"""
def _load_extension():
    global _EXT
    if _EXT is None:
        _EXT=load_inline(name='streamdsgn_smooth_contention_microchain_ext_v1',cpp_sources=_CPP_SRC,cuda_sources=_CUDA_SRC,functions=None,
                         extra_cuda_cflags=['-O3','--use_fast_math'],extra_cflags=['-O3'],with_cuda=True,
                         verbose=os.environ.get('SMOOTH_CONTENTION_BUILD_VERBOSE','0')=='1')
    return _EXT
def gpu_sm_count(device=None):
    if device is None: device=torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device).multi_processor_count)
class SmoothCudaContention:
    def __init__(self,strength,window_ms=100.0,slice_ms=0.25,start_delay_ms=2.0,threads=256,device=None):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
        self.device=torch.cuda.current_device() if device is None else int(device); torch.cuda.set_device(self.device)
        self.strength=float(strength); self.window_ms=float(window_ms); self.slice_ms=float(slice_ms); self.start_delay_ms=float(start_delay_ms); self.threads=int(threads)
        if self.strength<=0 or self.window_ms<=0 or self.slice_ms<=0: raise ValueError('strength/window/slice must be >0')
        if self.start_delay_ms<0 or self.start_delay_ms>=self.window_ms: raise ValueError('invalid start delay')
        self.sm_count=gpu_sm_count(self.device); self.blocks=max(1,int(math.ceil(self.strength*self.sm_count)))
        self.num_slices=max(1,int(math.ceil(self.window_ms/self.slice_ms))); self.nominal_chain_ms=self.num_slices*self.slice_ms
        least_priority, greatest_priority = torch.cuda.get_stream_priority_range()
        # CUDA/PyTorch convention:
        #   least_priority   -> lowest-priority scheduling class
        #   greatest_priority -> highest-priority scheduling class
        # Keep contention LOW priority so detector work can be admitted at
        # every short-kernel boundary.
        self.background_priority = int(least_priority)
        self.detector_priority_hint = int(greatest_priority)
        self.stream=torch.cuda.Stream(
            device=self.device,
            priority=self.background_priority,
        )
        self.sink=torch.zeros(
            1,
            device=f'cuda:{self.device}',
            dtype=torch.float32,
        )
        self.ext=_load_extension()
        self.start_event=None; self.end_event=None
    def launch(self):
        self.start_event=torch.cuda.Event(enable_timing=True); self.end_event=torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            self.start_event.record(); self.ext.launch_smooth_contention_chain(self.sink,self.blocks,self.threads,self.slice_ms,self.num_slices); self.end_event.record()
        self.start_event.synchronize()
        if self.start_delay_ms>0: time.sleep(self.start_delay_ms/1000.0)
    def finish(self):
        if self.end_event is None: return 0.0
        self.end_event.synchronize(); return float(self.start_event.elapsed_time(self.end_event))
    @property
    def available_ms(self): return self.nominal_chain_ms-self.start_delay_ms-max(2*self.slice_ms,1.0)
    def assert_covers_forward(self,forward_ms,margin_ms=2.0):
        if float(forward_ms)+float(margin_ms)>self.available_ms:
            raise RuntimeError(f'contention chain did not cover forward: forward={forward_ms:.3f} ms, available={self.available_ms:.3f} ms')
    def config(self):
        return {'type':'short_kernel_microchain_compute_contention','strength':self.strength,'sm_count':self.sm_count,'blocks':self.blocks,'threads':self.threads,
                'window_ms':self.window_ms,'slice_ms':self.slice_ms,'num_slices':self.num_slices,'nominal_chain_ms':self.nominal_chain_ms,
                'start_delay_ms':self.start_delay_ms,
                'background_stream_priority':self.background_priority,
                'detector_stream_priority_hint':self.detector_priority_hint,
                'strength_semantics':'nominal blocks/SM; actual level is fixed-prefix measured latency'}
def stream_priority_range():
    return tuple(int(x) for x in torch.cuda.get_stream_priority_range())

def make_high_priority_detector_stream(device=None):
    if device is None:
        device = torch.cuda.current_device()
    least_priority, greatest_priority = stream_priority_range()
    return torch.cuda.Stream(
        device=int(device),
        priority=int(greatest_priority),
    )

def build_extension_only():
    ext=_load_extension()
    print('[OK] smooth contention extension loaded:',ext)
    print('[GPU] SM count:',gpu_sm_count())
    print('[CUDA] stream priority range (least, greatest):', stream_priority_range())
if __name__=='__main__': build_extension_only()
