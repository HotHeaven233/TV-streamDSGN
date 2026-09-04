#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from pcdet.config import cfg,cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import extract_fixed_layer1_prefix
from test_stream_buffer_timestamp import build_scene_index,load_one
from smooth_cuda_contention import SmoothCudaContention, make_high_priority_detector_stream, stream_priority_range, stream_priority_value
torch.backends.cudnn.benchmark=True
def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--full_cfg',required=True); p.add_argument('--full_ckpt',required=True); p.add_argument('--warmup',type=int,default=10); p.add_argument('--frames',type=int,default=80); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--contention-strength',type=float,default=0.0); p.add_argument('--contention-window-ms',type=float,default=100.0); p.add_argument('--contention-slice-ms',type=float,default=0.25); p.add_argument('--contention-start-delay-ms',type=float,default=2.0); p.add_argument('--contention-threads',type=int,default=256); p.add_argument('--output_json',required=True); return p.parse_args()
def make_cfg(path):
    cfg_from_yaml_file(path,cfg); cfg.TAG=Path(path).stem; cfg.DATA_CONFIG.INFER_TIME_PATH=None
    if hasattr(cfg.MODEL,'BACKBONE_3D'): cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained=None
    return cfg
def stat(v):
    x=np.asarray(v,dtype=np.float64); return {'n':int(x.size),'mean_ms':float(x.mean()),'std_ms':float(x.std()),'p10_ms':float(np.percentile(x,10)),'p50_ms':float(np.percentile(x,50)),'p90_ms':float(np.percentile(x,90)),'p99_ms':float(np.percentile(x,99)),'min_ms':float(x.min()),'max_ms':float(x.max())}
def choose(ds,n):
    scene,idx=sorted(build_scene_index(ds).items(),key=lambda kv:len(kv[1]),reverse=True)[0]; out=[]
    while len(out)<n: out.extend(idx)
    return scene,out[:n]
def one_probe(backbone,frame,amp,model_stream,ready):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(model_stream):
        model_stream.wait_event(ready); s.record()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
            _=extract_fixed_layer1_prefix(backbone,frame)
        e.record()
    e.synchronize(); return float(s.elapsed_time(e))
def main():
    a=parse_args(); c=make_cfg(a.full_cfg); logger=common_utils.create_logger(); ds,_,_=build_dataloader(dataset_cfg=c.DATA_CONFIG,class_names=c.CLASS_NAMES,batch_size=1,dist=False,workers=a.workers,logger=logger,training=False)
    model=build_network(model_cfg=c.MODEL,num_class=len(c.CLASS_NAMES),dataset=ds); model.load_params_from_file(filename=a.full_ckpt,logger=logger,to_cpu=True); model.cuda().eval(); amp=bool(model.use_amp_dict['TEST']); model_stream=make_high_priority_detector_stream(torch.cuda.current_device())
    print('[CUDA] stream priority range (least, greatest):', stream_priority_range())
    print('[CUDA] detector stream priority:', stream_priority_value(model_stream))
    scene,idx=choose(ds,a.warmup+a.frames)
    cont=None
    if a.contention_strength>0: cont=SmoothCudaContention(a.contention_strength,a.contention_window_ms,a.contention_slice_ms,a.contention_start_delay_ms,a.contention_threads,torch.cuda.current_device())
    def run(i):
        batch=load_one(ds,i); ready=torch.cuda.Event(enable_timing=False); ready.record(torch.cuda.current_stream())
        if cont: cont.launch()
        ms=one_probe(
            model.backbone_3d,
            batch['token'],
            amp,
            model_stream,
            ready,
        )
        cms=0.0
        if cont:
            cms=cont.finish()
            cont.assert_covers_forward(ms,1.0)
        return ms,cms
    for i in idx[:a.warmup]: run(i)
    vals=[]; chains=[]
    for i in idx[a.warmup:]:
        m,cms=run(i); vals.append(m); chains.append(cms) if cont else None
    res={'scope':'fixed Stem+Layer1 only, CUDA-event GPU time','scene':scene,'warmup':a.warmup,'frames':a.frames,'amp_enabled':amp,'contention':None if cont is None else cont.config(),'probe':stat(vals),'contention_chain_actual':None if not chains else stat(chains),'raw_ms':vals}
    p=Path(a.output_json); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(res,indent=2)+'\n'); compact=dict(res); compact.pop('raw_ms'); print(json.dumps(compact,indent=2))
if __name__=='__main__': main()
