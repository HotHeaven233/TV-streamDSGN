import torch
import copy
import time
from collections import deque
from torch.cuda.amp import autocast as autocast
# from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .stream_detector3d_template import StreamDetector3DTemplate


class STREAM(StreamDetector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.history_tag = model_cfg.get('HISTORY_TAG', None)  # prev2, prev
        self.history_features_name = model_cfg.get(
            'HISTORY_FEATURES_NAME', None)
        self.history_feature_queue = deque(maxlen=len(
            self.history_tag)) if self.history_tag is not None else None
        self.use_kd = model_cfg.get('USE_KD', None)
        self.save_time = model_cfg.get('SAVE_TIME', False)

        feature_extractor_name = model_cfg.get('FEATURE_EXTRACTOR', None)
        assert feature_extractor_name is not None, 'FEATURE_EXTRACTOR should be a module name list!'
        self.feature_extractor = []
        fusion_stage_name = model_cfg.get('FUSION_STAGE', None)
        assert fusion_stage_name is not None, 'FUSION_STAGE should be a module name list!'
        self.fusion_module = []
        after_fusion_name = model_cfg.get('AFTER_FUSION', None)
        assert after_fusion_name is not None, 'AFTER_FUSION should be a module name list!'
        self.after_fusion_blocks = []

        for cur_module in self.module_list:
            if type(cur_module).__name__ in feature_extractor_name:
                self.feature_extractor.append(cur_module)
            if type(cur_module).__name__ in fusion_stage_name:
                self.fusion_module.append(cur_module)
            if type(cur_module).__name__ in after_fusion_name:
                self.after_fusion_blocks.append(cur_module)

    # def create_module_dict(self):
    #     result = {}
    #     count_dict = {}
    #     for i in range(len(self.module_name_list)):
    #         if self.module_name_list[i] in count_dict:
    #             count_dict[self.module_name_list[i]] += 1
    #             result[self.module_name_list[i] +
    #                    str(count_dict[self.module_name_list[i]])] = self.module_list[i]
    #         else:
    #             count_dict[self.module_name_list[i]] = 0
    #             result[self.module_name_list[i]] = self.module_list[i]
    #     return result
    def cuda(self, device=None):
        model = super().cuda(device)
        for i in range(len(self.fusion_module)):
            if hasattr(self.fusion_module[i], 'cuda'):
                self.fusion_module[i] = self.fusion_module[i].cuda(device)
        return model

    def obtain_history_feature(self, batch_dict):
        """
        Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        if self.mode == 'TRAIN':
            if self.with_grad_when_obtaining_history:
                for cur_module in self.feature_extractor:
                    batch_dict = cur_module(batch_dict)
                history_feature = {}
                for key in self.history_features_name:
                    history_feature[key] = batch_dict[key]

            else:
                with torch.no_grad():
                    for cur_module in self.feature_extractor:
                        batch_dict = cur_module(batch_dict)
                    history_feature = {}
                    for key in self.history_features_name:
                        history_feature[key] = copy.deepcopy(batch_dict[key])
            self.history_feature_queue.append(
                (batch_dict['this_sample_idx'], history_feature))

        # elif self.mode == 'TEST':
        #     if batch_dict['prev_sample_idx'] == '':  # The scene changes
        #         self.history_feature_queue.clear()
        #     if batch_dict.get('history_features', None):
        #         self.history_feature_queue.append(
        #             batch_dict['history_features'])
        else:
            raise NotImplementedError

    def obtain_future_feature(self, batch_dict):
        """
        Obtain future BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        if self.mode == 'TRAIN':
            if self.use_kd.with_grad:
                for cur_module in self.feature_extractor:
                    batch_dict = cur_module(batch_dict)
                # future_feature = {
                #     k: v for k, v in batch_dict.items() if k in self.use_kd.feature_name}
                future_feature = batch_dict[self.use_kd.feature_name[0]]  # only support one feature
            else:
                with torch.no_grad():
                    for cur_module in self.feature_extractor:
                        batch_dict = cur_module(batch_dict)
                    # future_feature = {k: copy.deepcopy(
                    #     v) for k, v in batch_dict.items() if k in self.use_kd.feature_name}
                    # future_feature = copy.deepcopy(batch_dict[self.use_kd.feature_name[0]])
                    future_feature = batch_dict[self.use_kd.feature_name[0]].detach()
            return future_feature
        else:
            raise NotImplementedError

    def pop_key_in_history_input(self, batch_dict):
        keys_to_remove = ['left_img',
                          'right_img',
                          'points',
                          'voxels',
                          'voxel_coords',
                          'voxel_num_points',
                          'input_depth_gt_img',
                          'depth_gt_img',
                          'depth_fgmask_img',
                          ]
        for k in keys_to_remove:
            if k in batch_dict.keys():
                batch_dict.pop(k)

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss = 0.
        tb_dict = {}

        # MTD_HEAD_ONLY_LOSS_BEGIN
        if getattr(
            self,
            '_train_mtd_head_only',
            False
        ):
            if getattr(
                self,
                'dense_head',
                None
            ) is None:
                raise RuntimeError(
                    'MTD head-only training '
                    'requires dense_head'
                )

            loss_rpn, tb_dict = (
                self.dense_head.get_loss(
                    batch_dict,
                    tb_dict
                )
            )

            tb_dict.update(
                loss_rpn=loss_rpn.item()
            )

            return (
                loss_rpn,
                tb_dict,
                disp_dict,
            )
        # MTD_HEAD_ONLY_LOSS_END

        if getattr(self, 'dense_head_2d', None):
            loss_rpn_2d, tb_dict = self.dense_head_2d.get_loss(
                batch_dict, tb_dict)
            tb_dict['loss_rpn2d'] = loss_rpn_2d.item()
            loss += loss_rpn_2d

        if getattr(self, 'dense_head', None):
            loss_rpn, tb_dict = self.dense_head.get_loss(batch_dict, tb_dict)
            loss += loss_rpn
            tb_dict.update(loss_rpn=loss_rpn.item())

        if (not self.model_cfg.BACKBONE_3D.get('drop_psv', False) and not self.model_cfg.BACKBONE_3D.get('drop_psv_loss', False)) or self.model_cfg.BACKBONE_3D.get('front_surface_depth', False):
            loss_depth, tb_dict = self.depth_loss_head.get_loss(
                batch_dict, tb_dict)
            tb_dict.update(loss_depth=loss_depth.item())
            if torch.isnan(loss_depth):
                loss += sum([i.sum()
                            for i in batch_dict['token']['depth_preds']]) * 0.
                print('-------------- NaN depth loss')
            else:
                loss += loss_depth
        return loss, tb_dict, disp_dict

    def forward(self, batch_dict):
        with autocast(enabled=self.use_amp_dict[self.mode]):
            if self.mode == 'TRAIN':
                return self.forward_train(batch_dict)
            if self.mode == 'TEST':
                if self.save_time:
                    return self.forward_test_save_time(batch_dict)
                else:
                    return self.forward_test(batch_dict)

    def forward_train(self, batch_dict):
        cur_data = batch_dict['token']

        for cur_module in self.feature_extractor:
            cur_data = cur_module(cur_data)

        if self.history_tag is not None:
            for tag in self.history_tag:  # prev2, prev
                # The first frame has no history information
                if batch_dict[tag] is not None:
                    self.obtain_history_feature(batch_dict[tag])
                    # self.pop_key_in_history_input(batch_dict[tag])
        cur_data['history_features'] = self.history_feature_queue
        if self.use_kd is not None and self.training:
            if batch_dict['next'] is not None:
                cur_data['teacher_feature'] = self.obtain_future_feature(
                    batch_dict['next'])

        for cur_module in self.fusion_module:
            cur_data = cur_module(cur_data)

        for cur_module in self.after_fusion_blocks:
            cur_data = cur_module(cur_data)

        batch_dict['token'] = cur_data
        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
        ret_dict = {
            'loss': loss
        }

        if self.history_tag is not None:
            self.history_feature_queue.clear()

        return ret_dict, tb_dict, disp_dict

    def forward_test(self, batch_dict):
        cur_data = batch_dict['token']

        # The scene changes
        if self.history_tag is not None and ('prev_sample_idx' not in cur_data or cur_data['prev_sample_idx'] == ''):
            self.history_feature_queue.clear()

        for cur_module in self.feature_extractor:
            cur_data = cur_module(cur_data)

        cur_data['history_features'] = self.history_feature_queue
        if self.history_tag is not None:
            history_features = {}
            for feature_name in self.history_features_name:
                history_features[feature_name] = cur_data[feature_name].clone()

        for cur_module in self.fusion_module:
            cur_data = cur_module(cur_data)

        for cur_module in self.after_fusion_blocks:
            cur_data = cur_module(cur_data)
    
        pred_dicts, ret_dicts = self.post_processing(cur_data)
        
        if self.history_tag is not None:
            # Send the current frame to the history queue as the previous frame of the next frame
            self.history_feature_queue.append(
                (cur_data['this_sample_idx'], history_features))

        for k in cur_data.keys():
            if k.startswith('depth_error_'):
                if isinstance(cur_data[k], list):
                    ret_dicts[k] = cur_data[k]
                elif len(cur_data[k].shape) == 0:
                    ret_dicts[k] = cur_data[k].item()

        if getattr(self, 'dense_head_2d', None) and 'boxes_2d_pred' in cur_data:
            assert len(pred_dicts) == len(cur_data['boxes_2d_pred'])
            for pred_dict, pred_2d_dict in zip(pred_dicts, cur_data['boxes_2d_pred']):
                pred_dict['pred_boxes_2d'] = pred_2d_dict['pred_boxes_2d']
                pred_dict['pred_scores_2d'] = pred_2d_dict['pred_scores_2d']
                pred_dict['pred_labels_2d'] = pred_2d_dict['pred_labels_2d']

        return pred_dicts, ret_dicts
    
    def forward_test_save_time(self, batch_dict):
        cur_data = batch_dict['token']
        spend_time = {}

        # The scene changes
        if self.history_tag is not None and ('prev_sample_idx' not in cur_data or cur_data['prev_sample_idx'] == ''):
            self.history_feature_queue.clear()

        for cur_module in self.feature_extractor:
            torch.cuda.synchronize()
            t1 = time.time()
            cur_data = cur_module(cur_data)
            torch.cuda.synchronize()
            t2 = time.time()
            spend_time[type(cur_module).__name__] = (t2 - t1) * 1000

        torch.cuda.synchronize()
        t1 = time.time()
        cur_data['history_features'] = self.history_feature_queue
        if self.history_tag is not None:
            history_features = {}
            for feature_name in self.history_features_name:
                history_features[feature_name] = cur_data[feature_name].clone()

        for cur_module in self.fusion_module:
            cur_data = cur_module(cur_data)
        t2 = time.time()
        spend_time[type(cur_module).__name__] = (t2 - t1) * 1000

        for cur_module in self.after_fusion_blocks:
            torch.cuda.synchronize()
            t1 = time.time()
            cur_data = cur_module(cur_data)
            t2 = time.time()
            spend_time[type(cur_module).__name__] = (t2 - t1) * 1000

        torch.cuda.synchronize()
        t1 = time.time()
        pred_dicts, ret_dicts = self.post_processing(cur_data)
        if self.history_tag is not None:
            # Send the current frame to the history queue as the previous frame of the next frame
            self.history_feature_queue.append(
                (cur_data['this_sample_idx'], history_features))

        for k in cur_data.keys():
            if k.startswith('depth_error_'):
                if isinstance(cur_data[k], list):
                    ret_dicts[k] = cur_data[k]
                elif len(cur_data[k].shape) == 0:
                    ret_dicts[k] = cur_data[k].item()

        if getattr(self, 'dense_head_2d', None) and 'boxes_2d_pred' in cur_data:
            assert len(pred_dicts) == len(cur_data['boxes_2d_pred'])
            for pred_dict, pred_2d_dict in zip(pred_dicts, cur_data['boxes_2d_pred']):
                pred_dict['pred_boxes_2d'] = pred_2d_dict['pred_boxes_2d']
                pred_dict['pred_scores_2d'] = pred_2d_dict['pred_scores_2d']
                pred_dict['pred_labels_2d'] = pred_2d_dict['pred_labels_2d']
        t2 = time.time()
        spend_time['post_processing'] = (t2 - t1) * 1000
        ret_dicts['spend_time'] = spend_time
        return pred_dicts, ret_dicts
