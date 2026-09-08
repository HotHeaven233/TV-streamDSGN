from .lasp_stream import LASP_STREAM
from .transtreaming_stream_v2 import TRANSTREAMING_STREAM_V2
from .transtreaming_stream import TRANSTREAMING_STREAM
from .stream import STREAM
import torch.distributed as dist
from pcdet.utils.common_utils import create_logger

__all__ = {
    'transtreaming_stream_v2': TRANSTREAMING_STREAM_V2,
    'transtreaming_stream': TRANSTREAMING_STREAM,
    'stream': STREAM,
    'stream_lasp': LASP_STREAM,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    try:
        logger = create_logger(rank=dist.get_rank())
    except:
        logger = create_logger()
    if hasattr(model_cfg, 'PRETRAINED_MODEL') and model_cfg.PRETRAINED_MODEL:
        model.load_params_from_file(
            filename=model_cfg.PRETRAINED_MODEL, to_cpu=True, logger=logger)

    return model
