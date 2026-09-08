from collections import namedtuple

import numpy as np
import torch

def build_network(model_cfg, num_class, dataset):
    # Import only the detector family actually requested.
    # This avoids importing unrelated LiDAR/spconv modules.
    if model_cfg['NAME'].startswith('stereo'):
        from .detectors_stereo import build_detector
    elif (
        model_cfg['NAME'].startswith('stream')
        or
        model_cfg['NAME'].startswith('transtreaming')
    ):
        from .detectors_stream import build_detector
    else:
        from .detectors_lidar import build_detector

    return build_detector(
        model_cfg=model_cfg,
        num_class=num_class,
        dataset=dataset
    )

def load_data_to_gpu(batch_dict):
    def process_single_frame(single_dict):
        # The first frame has no history information
        if single_dict is not None:
            for key, val in single_dict.items():
                if not isinstance(val, np.ndarray):
                    continue
                if key in ['scene', 'this_sample_idx', 'prev2_sample_idx', 'prev_sample_idx', 'next_sample_idx', 'frame_id', 'metadata', 'calib', 'calib_ori', 'image_shape', 'gt_names']:
                    continue
                if val.dtype in [np.float32, np.float64]:
                    single_dict[key] = torch.from_numpy(val).float().cuda()
                elif val.dtype in [np.uint8, np.int32, np.int64]:
                    single_dict[key] = torch.from_numpy(val).long().cuda()
                elif val.dtype in [bool]:
                    pass
                else:
                    raise ValueError(f"invalid data type {key}: {type(val)}")

    if batch_dict.get('token', False):
        for _, single_dict in batch_dict.items():
            process_single_frame(single_dict)
    else:
        process_single_frame(batch_dict)


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
