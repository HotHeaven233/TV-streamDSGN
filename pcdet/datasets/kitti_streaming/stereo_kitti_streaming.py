import copy
import json
import pickle
import numpy as np
import torch
from skimage import io
from easydict import EasyDict
from scipy.optimize import linear_sum_assignment
import concurrent.futures as futures

from pcdet.utils import box_utils, calibration_kitti_tracking, common_utils, tracking3d_kitti, depth_map_utils
from pcdet.datasets.stereo_streaming_template import StereoStreamingTemplate



class StereoKittiStreaming(StereoStreamingTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        assert getattr(self.dataset_cfg, 'later_flip', True)
        self.boxes_gt_in_cam2_view = getattr(
            self.dataset_cfg, 'BOXES_GT_IN_CAM2_VIEW', False)
        assert not self.boxes_gt_in_cam2_view
        self.use_van = self.dataset_cfg.USE_VAN and training
        self.use_person_sitting = self.dataset_cfg.USE_PERSON_SITTING and training
        self.cat_reflect = self.dataset_cfg.CAT_REFLECT_DIM
        assert not self.cat_reflect  # remove reflection

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')

        self.type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
        split_dir = self.prepare_data_path / \
            'ImageSets' / (self.split + '.json')
        assert split_dir.exists()
        with open(split_dir, 'r') as f:
            self.sample_id_list = json.load(f)
        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        self._maybe_add_mtd_future_tags()

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.prepare_data_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        assert len(self.sample_id_list) == len(self.kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' %
                             (len(kitti_infos)))

    def _maybe_add_mtd_future_tags(self):
        """
        Add next2 / next3 metadata for MTD head training.
        The future chain is followed strictly inside one scene.
        """
        steps = self.dataset_cfg.get(
            'MTD_FUTURE_STEPS',
            []
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
                'MTD_FUTURE_STEPS must '
                f'contain integers >= 2: {steps}'
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
                    ''
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
                    f'duplicate MTD token: {key}'
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
                next_id = (
                    cur_info[
                        'sample_idx'
                    ][
                        'frame_tag'
                    ].get(
                        'next',
                        ''
                    )
                )

                if (
                    next_id is None
                    or
                    str(next_id) == ''
                ):
                    break

                next_key = (
                    scene,
                    str(next_id),
                )

                next_info = lookup.get(
                    next_key,
                    None,
                )

                if next_info is None:
                    break

                if step in steps:
                    tag = (
                        f'next{step}'
                    )

                    frame_tags[
                        tag
                    ] = str(
                        next_id
                    )

                    item[
                        'infos'
                    ][
                        tag
                    ] = copy.deepcopy(
                        next_info[
                            'infos'
                        ][
                            'token'
                        ]
                    )

                    valid_count[
                        step
                    ] += 1

                cur_info = next_info

            for step in steps:
                tag = (
                    f'next{step}'
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
                'MTD future tags: '
                +
                ', '.join(
                    f'next{step}: '
                    f'valid={valid_count[step]}, '
                    f'invalid={invalid_count[step]}'
                    for step in steps
                )
            )

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')

        split_dir = self.prepare_data_path / \
            'ImageSets' / (self.split + '.json')
        assert split_dir.exists()
        with open(split_dir, 'r') as f:
            self.sample_id_list = json.load(f)

    def get_lidar(self, scene, filename):
        scene = scene.split('_')[0]
        lidar_file = self.root_split_path / 'velodyne' / \
            ('%s' % scene) / ('%s.bin' % filename)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, scene, filename):
        scene = scene.split('_')[0]
        img_file = self.root_split_path / 'image_02' / \
            ('%s' % scene) / ('%s.png' % filename)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_image(self, scene, filename, image_id=2):
        scene = scene.split('_')[0]
        img_file = self.root_split_path / \
            ('image_0%s' % image_id) / scene / ('%s.png' % filename)
        assert img_file.exists()
        return io.imread(img_file).copy()

    def get_label(self, scene, filename):
        scene = scene.split('_')[0]
        label_file = self.root_split_path / 'label_02' / ('%s.txt' % scene)
        assert label_file.exists()
        return tracking3d_kitti.get_objects_from_label(label_file, filename)

    def get_calib(self, scene):
        scene = scene.split('_')[0]
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % scene)
        assert calib_file.exists()
        return calibration_kitti_tracking.Calibration(calib_file)

    def get_road_plane(self, scene, filename):
        scene = scene.split('_')[0]
        plane_file = self.root_split_path / 'planes' / \
            scene / ('%s.txt' % filename)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(
            pts_img[:, 0] > 0, pts_img[:, 0] < img_shape[1] - 1)
        val_flag_2 = np.logical_and(
            pts_img[:, 1] > 0, pts_img[:, 1] < img_shape[0] - 1)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None, mode_2d=False):
        """
        Args:
        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib_aug = batch_dict['calib'][batch_index]
            calib_ori = batch_dict['calib_ori'][batch_index] if 'calib_ori' in batch_dict else calib_aug
            image_shape = batch_dict['image_shape'][batch_index]
            # NOTE: in stereo mode, the 3d boxes are predicted in pseudo lidar coordinates
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                pred_boxes, None, pseudo_lidar=True, pseduo_cam2_view=self.boxes_gt_in_cam2_view)
            # only for debug, calib.flipped should be False when testing
            if calib_aug.flipped:
                pred_boxes_camera = box_utils.boxes3d_fliplr(
                    pred_boxes_camera, cam_view=True)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib_ori, image_shape=image_shape,
                fix_neg_z_bug=True
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1],
                                             pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            return pred_dict

        def generate_single_2d_sample_dict(batch_index, box_dict):
            def to_numpy(x):
                if isinstance(x, np.ndarray):
                    return x
                elif isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                else:
                    raise ValueError('wrong type of input')
            pred_scores_2d = to_numpy(box_dict['pred_scores_2d'])
            pred_boxes_2d = to_numpy(box_dict['pred_boxes_2d'])
            pred_labels_2d = to_numpy(box_dict['pred_labels_2d'])
            pred_dict = get_template_prediction(pred_scores_2d.shape[0])
            calib = batch_dict['calib'][batch_index]
            # calib_ori = batch_dict['calib_ori'][batch_index] if 'calib_ori' in batch_dict else calib
            if pred_scores_2d.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels_2d - 1]
            pred_dict['bbox'] = pred_boxes_2d[:, :4]

            pred_dict['bbox'][:, [0, 2]] += calib.offsets[0]
            pred_dict['bbox'][:, [1, 3]] += calib.offsets[1]

            pred_dict['score'] = pred_scores_2d

            return pred_dict

        annos = []
        batch_dict = batch_dict['token']
        for index, box_dict in enumerate(pred_dicts):
            scene = batch_dict['scene'][index]
            frame_id = batch_dict['this_sample_idx'][index]
            next_sample_idx = batch_dict['next_sample_idx'][index] if batch_dict.get(
                'next_sample_idx', None) is not None else None

            if not mode_2d:
                single_pred_dict = generate_single_sample_dict(index, box_dict)
            else:
                single_pred_dict = generate_single_2d_sample_dict(
                    index, box_dict)
            single_pred_dict['scene'] = scene
            single_pred_dict['frame_id'] = frame_id
            single_pred_dict['next_frame_id'] = next_sample_idx
            annos.append(single_pred_dict)
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.8f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        return annos

    def load_input_data(self, info):
        '''
            load input data from file.
        '''
        scene = info['sample_idx']['scene']
        img_shape = info['infos']['token']['image_shape']
        prev_sample_idx = info['sample_idx']['frame_tag'].get('prev', None)
        prev2_sample_idx = info['sample_idx']['frame_tag'].get('prev2', None)
        next_sample_idx = info['sample_idx']['frame_tag'].get('next', None)

        if self.mode == 'train':
            frame_tag = info['sample_idx']['frame_tag']
        else:
            # We don’t need historical information in the inference stage
            # and the future information only serves as supervision during the training phase
            frame_tag = ['token']

        def load_single_frame_data(tag):
            ret_dict = {}
            this_sample_idx = info['sample_idx']['frame_tag'][tag]
            if this_sample_idx == '':
                ret_dict['frame_valid'] = False
                return tag, ret_dict

            ret_dict['frame_valid'] = True

            # load calib
            calib = self.get_calib(scene)
            calib_ori = copy.deepcopy(calib)

            # load images
            left_img = self.get_image(scene, this_sample_idx, 2)
            right_img = self.get_image(scene, this_sample_idx, 3)

            # get road plane
            road_plane = self.get_road_plane(scene, this_sample_idx)

            # get points
            raw_points = self.get_lidar(scene, this_sample_idx)
            pts_rect = calib.lidar_to_rect(raw_points[:, 0:3])
            reflect = raw_points[:, 3:4]
            if self.dataset_cfg.FOV_POINTS_ONLY:
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                pts_rect = pts_rect[fov_flag]
                reflect = reflect[fov_flag]
            if self.cat_reflect:
                input_points = np.concatenate(
                    [calib.rect_to_lidar_pseudo(pts_rect), reflect], 1)
            else:
                input_points = calib.rect_to_lidar_pseudo(pts_rect)
            ret_dict['input_data'] = {
                'scene': scene,
                'this_sample_idx': this_sample_idx,
                'calib': calib,
                'calib_ori': calib_ori,
                'left_img': left_img,
                'right_img': right_img,
                'image_shape': img_shape,
                'road_plane': road_plane,
                'points': input_points,
            }
            if prev2_sample_idx is not None:
                ret_dict['input_data'].update(
                    prev2_sample_idx=prev2_sample_idx)
            if prev_sample_idx is not None:
                ret_dict['input_data'].update(prev_sample_idx=prev_sample_idx)
            if next_sample_idx is not None:
                ret_dict['input_data'].update(next_sample_idx=next_sample_idx)

            # get gt_info
            annos = info['infos'][tag]['annos']
            if len(annos['name']) > 0:
                if self.use_van:
                    # Car 14357, Van 1297
                    annos['name'][annos['name'] == 'Van'] = 'Car'
                if self.use_person_sitting:
                    # Ped 2207, Person_sitting 56
                    annos['name'][annos['name'] ==
                                  'Person_sitting'] = 'Pedestrian'

                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                object_id = annos['object_id']
                index_in_label = annos['index_in_label']
                gt_boxes_camera = np.concatenate(
                    [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_annots_boxes_2d = annos['bbox']
                gt_truncated = annos['truncated']
                gt_occluded = annos['occluded']
                gt_difficulty = annos['difficulty']
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                    gt_boxes_camera, calib, pseudo_lidar=True, pseudo_cam2_view=self.boxes_gt_in_cam2_view)
                gt_boxes_mask = np.array(
                    [n in self.class_names for n in gt_names], dtype=np.bool_)
                ret_dict['gt_info'] = {
                    'gt_valid': True,
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar,
                    'gt_annots_boxes_2d': gt_annots_boxes_2d,
                    'gt_truncated': gt_truncated,
                    'gt_occluded': gt_occluded,
                    'gt_difficulty': gt_difficulty,
                    'gt_index': index_in_label,
                    'gt_boxes_mask': gt_boxes_mask,
                    'object_id': object_id,  # dtype = str
                }
            else:
                ret_dict['gt_info'] = {
                    'gt_valid': False,
                    'gt_names': np.array([], dtype=str),
                    'gt_boxes': np.array([], dtype=float).reshape(-1, 7),
                    'gt_annots_boxes_2d': np.array([], dtype=float).reshape(-1, 4),
                    'gt_truncated': np.array([], dtype=float).reshape(-1),
                    'gt_occluded': np.array([], dtype=float).reshape(-1),
                    'gt_difficulty': np.array([], dtype=np.int32).reshape(-1),
                    'gt_index': np.array([], dtype=np.int32).reshape(-1),
                    'gt_boxes_mask':  np.array([], dtype=np.bool_),
                    'object_id': np.array([], dtype=str),
                }
            return tag, ret_dict

        with futures.ThreadPoolExecutor(4) as executor:
            ret_infos = executor.map(load_single_frame_data, frame_tag)

        # token: info; prev2:info; ...
        ret_infos = {key: value for key, value in list(ret_infos)}
        return ret_infos

    def streaming_simulation(self, gt_annos, det_annos, sp_strategy='copy', use_velo=False):
        assert self.empirical is not None, 'self.empirical is None, this will not allow streaming simulation'
        # annos_frequency is consistent with input_frequcency in kitti
        if sp_strategy == 'kf':
            from .kalman_filter import KalmanFilter
            kfilter = KalmanFilter(R_fac=40)
        
        if sp_strategy == 'tracker':
            try:
                from tracker import tracker
            except:
                raise ImportError('Please install tracker from https://github.com/hailanyi/3D-Multi-Object-Tracker')

        def get_matching(curr, last, curr_cls, last_cls):
            # curr: [M, 5]
            # last: [N, 5]
            # curr_cls: [M]
            # last_cls: [N]
            dist_thres = 10
            M = curr.shape[0]
            N = last.shape[0]

            curr_stack = curr[:, None, :3].repeat(N, 1)
            last_stack = last[None, :, :3].repeat(M, 0)
            affinity_mat = np.linalg.norm(curr_stack - last_stack, axis=2)

            curr_cls_stack = curr_cls[:, None].repeat(N, 1)
            last_cls_stack = last_cls[None, :].repeat(M, 0)
            affinity_mask = (curr_cls_stack == last_cls_stack)

            affinity_mat[~affinity_mask] = 1e6
            idx1, idx2 = linear_sum_assignment(affinity_mat)
            keep_idx1 = []
            keep_idx2 = []
            for i1, i2 in zip(idx1, idx2):
                if affinity_mat[i1, i2] < dist_thres:
                    keep_idx1.append(i1)
                    keep_idx2.append(i2)

            keep_idx1 = np.array(keep_idx1)
            keep_idx2 = np.array(keep_idx2)
            return keep_idx1, keep_idx2

        def modify_token(data, new_token):
            new_datas = copy.deepcopy(data)
            if len(new_datas) > 0:
                new_datas['sample_token'] = new_token
            return new_datas

        def kf_interpolation(kfilter, data, last, new_token, time_delta, time_consume=0, use_velo=False):
            '''
            refer from ASAP[CVPR2023]
            github url: https://github.com/JeffWang987/ASAP
            '''
            new_datas = copy.deepcopy(data)
            last_datas = copy.deepcopy(last)
            # Zt: [x, y, z, vx, vy]
            # cls: [M, 1] label
            if use_velo:
                if len(new_datas) > 0 and len(new_datas['name']) > 0:
                    location = copy.deepcopy(new_datas['location'])
                    vectorized_func = np.vectorize(self.type_to_id.get)
                    cls_name = vectorized_func(new_datas['name']).astype(np.float32)
                    if last_datas is not None and len(last_datas) > 0 and len(last_datas['name']) > 0:
                        last_location = copy.deepcopy(last_datas['location'])
                        last_cls_name = vectorized_func(last_datas['name']).astype(np.float32)
                        idx1, idx2 = get_matching(location, last_location, cls_name, last_cls_name)  # get matching for calculating velocity
                        if len(idx1) > 0:
                            location_selected = location[idx1]
                            last_location_selected = last_location[idx2]

                            vx = ((location_selected[:, 0] - last_location_selected[:, 0]) / (time_consume / 1000))
                            vy = ((location_selected[:, 1] - last_location_selected[:, 1]) / (time_consume / 1000))

                            new_location = np.zeros((location.shape[0], location.shape[1] + 2))
                            new_location[:, :3] = location
                            new_location[idx1, 3] = vx
                            new_location[idx1, 4] = vy
                            Zt = new_location
                        else:
                            vx = np.zeros(location.shape[0])
                            vy = np.zeros(location.shape[0])
                            Zt = np.concatenate((location, vx[..., np.newaxis], vy[..., np.newaxis]), axis=1)
                    else:
                        vx = np.zeros(location.shape[0])
                        vy = np.zeros(location.shape[0])
                        Zt = np.concatenate((location, vx[..., np.newaxis], vy[..., np.newaxis]), axis=1)

                    kf_X = kfilter(np.array(Zt).reshape(-1, 5),
                                np.array(cls_name), time_delta)  # M, 5
                    kf_velo = kf_X[..., -2:]
                    new_datas['sample_token'] = new_token
                    new_datas['location'] = kf_X[..., :3]
                    new_datas['location'][..., :2] = new_datas['location'][..., :2] + kf_velo * (time_delta / 1000)
                return new_datas

            else:
                if len(new_datas) > 0 and len(new_datas['name']) > 0:
                    location = copy.deepcopy(new_datas['location'])
                    rotation_y = np.concatenate((np.cos(new_datas['rotation_y'])[..., np.newaxis], np.sin(
                        new_datas['rotation_y'][..., np.newaxis])), axis=1)
                    Zt = np.concatenate((location, rotation_y), axis=1)
                    vectorized_func = np.vectorize(self.type_to_id.get)
                    cls_name = vectorized_func(new_datas['name']).astype(np.float32)
                    kf_X = kfilter(np.array(Zt).reshape(-1, 5),
                                np.array(cls_name), time_delta)  # M, 5
                    new_datas['sample_token'] = new_token
                    new_datas['location'] = kf_X[..., :3]
                    new_datas['rotation_y'] = np.arctan2(kf_X[..., 4], kf_X[..., 3])
                return new_datas
        
        def tracker_interpolation(tracker3d, data, last, new_token, time_delta, time_consume=0, count_idx=0):
            new_datas = copy.deepcopy(data)
            last_datas = copy.deepcopy(last)
            if len(new_datas) > 0 and len(new_datas['name']) > 0:
                det_scores = new_datas['score']
                box3d = new_datas['boxes_lidar']
                mask = det_scores > tracker3d.config.input_score
                det_scores = det_scores[mask]
                box3d = box3d[mask]
                bbs, bbs_id = tracker3d.tracking(box3d, scores=det_scores, timestamp=count_idx)
                count_idx += 1
                data['bbs_id'] = bbs_id
                location = copy.deepcopy(new_datas['location'])
                if last_datas is not None and len(last_datas) > 0 and len(last_datas['name']) > 0:
                    last_location = copy.deepcopy(last_datas['location'])
                    last_id = last_datas['bbs_id']
                    duplicate_ids = np.intersect1d(bbs_id, last_id)

                    # if len(duplicate_ids) > 0:
                    #     loc = []
                    #     loc_last = []
                    #     for id in duplicate_ids:
                    #         idx = np.where(bbs_id == id)[0]
                    #         last_idx = np.where(last_id == id)[0]
                    #         loc.append(location[idx])
                    #         loc_last.append(last_location[last_idx])
                    loc = []
                    loc_last = []
                    loc_indices = []

                    for idx, id in enumerate(bbs_id):
                        if id in duplicate_ids:
                            idx = np.where(bbs_id == id)[0]
                            loc.append(location[idx])
                            loc_indices.append(idx[0])
                            last_idx = np.where(last_id == id)[0]
                            loc_last.append(last_location[last_idx])
                    if len(duplicate_ids) > 0:
                        loc = np.concatenate(loc, axis=0)
                        loc_last = np.concatenate(loc_last, axis=0)
                        vx = ((loc[:, 0] - loc_last[:, 0]) / (time_consume / 1000))
                        vy = ((loc[:, 1] - loc_last[:, 1]) / (time_consume / 1000))
                        loc[:, 0] = loc[:, 0] + vx * (time_delta / 1000)
                        loc[:, 1] = loc[:, 1] + vy * (time_delta / 1000)
                        # velo = np.sqrt(vx ** 2 + vy ** 2)
                        # loc = loc + velo * (time_delta / 1000)
                        location[loc_indices] = loc

                    new_datas['location'] = location
                new_datas['sample_token'] = new_token
            return new_datas, count_idx


        def make_annos_correspond(gt_annos_with_frame_id, rst_infos):
            det_tmp_dict = {
                'name': np.array([]), 'truncated': np.array([]),
                'occluded': np.array([]), 'alpha': np.array([]),
                'bbox': np.array([]).reshape(-1, 4), 'dimensions': np.array([]).reshape(-1, 3),
                'location': np.array([]).reshape(-1, 3), 'rotation_y': np.array([]),
                'score': np.array([]), 'boxes_lidar': np.array([]).reshape(-1, 7),
            }
            new_gt_annos = []
            new_det_annos = []

            for k, v in rst_infos.items():
                new_gt_annos.append(gt_annos_with_frame_id[k])
                if len(v) > 0:
                    new_det_annos.append(v)
                else:
                    new_det_annos.append(det_tmp_dict)
            return new_gt_annos, new_det_annos

        ann_time_itv = 1000 / self.annos_frequency
        time_dist = self.empirical

        det_annos_with_scene = {}
        for x in det_annos:
            if x['scene'] in det_annos_with_scene.keys():
                det_annos_with_scene[x['scene']].append(x)
            else:
                det_annos_with_scene[x['scene']] = [x]

        # this token: next token
        input_token_sequence = {'{}_{}'.format(x['scene'], x['frame_id']): '{}_{}'.format(
            x['scene'], x['next_frame_id']) for x in det_annos}
        det_annos_with_frame_id = {'{}_{}'.format(
            x['scene'], x['frame_id']): x for x in det_annos}
        gt_annos_with_frame_id = {'{}_{}'.format(
            x['scene'], x['frame_id']): x for x in gt_annos}
        new_rst = {}

        for scene_id, this_det_annos in det_annos_with_scene.items():
            if sp_strategy == 'kf':
                kfilter.reset()
            elif sp_strategy == 'tracker':
                # same as kf
                config = {
                    'state_func_covariance': 100,
                    'measure_func_covariance': 0.001,
                    'prediction_score_decay': 0.03,
                    'LiDAR_scanning_frequency': 10,
                    'max_prediction_num': 12,
                    'max_prediction_num_for_new_object': 2,
                    'input_score': 0,
                    'init_score': 0.5,
                    'update_score': 0,
                    'post_score': 0.5,
                }
                config = EasyDict(config)
                tracker3d = tracker.Tracker3D(box_type="Kitti", tracking_features=False, config = config)

            time_elapsed = 0
            time_input = 0
            this_val_preds = []
            # first sample token
            this_sample_token = '{}_{}'.format(
                this_det_annos[0]['scene'], this_det_annos[0]['frame_id'])
            while True:
                this_model_time = time_dist.draw()
                time_elapsed = time_elapsed + this_model_time
                this_rst = det_annos_with_frame_id[this_sample_token]
                this_rst = {'sample_token': '{}_{}'.format(
                    this_rst['scene'], this_rst['frame_id']), **this_rst}
                this_pred_info = {
                    'time_input': time_input, 'results': this_rst}
                # time_elapsed: output time of the current sample
                # this_rst: most recent prediction
                this_val_preds.append({time_elapsed: this_pred_info})
                time_input += ann_time_itv
                this_sample_token = input_token_sequence[this_sample_token]

                if this_sample_token.rsplit('_', 1)[-1] == '':
                    break

                if time_elapsed < time_input:
                    time_elapsed = time_input

                elif time_elapsed >= time_input:
                    while time_elapsed >= time_input + ann_time_itv:
                        time_input += ann_time_itv
                        if input_token_sequence[this_sample_token].rsplit('_', 1)[-1] != '':
                            this_sample_token = input_token_sequence[this_sample_token]
                        else:
                            this_sample_token = this_sample_token

            time_ann = 0
            match_idx = 0
            count_idx = 0
            # first sample token
            this_sample_token = '{}_{}'.format(
                this_det_annos[0]['scene'], this_det_annos[0]['frame_id'])
            last_rst = {}
            this_rst = {}
            this_input_time = 0
            last_input_time = 0
            this_time_consume = 0
            while True:
                inf_time = next(iter(this_val_preds[match_idx].keys()))
                if time_ann < inf_time:
                    if sp_strategy == 'copy':
                        new_rst[this_sample_token] = modify_token(
                            this_rst, this_sample_token)
                    elif sp_strategy == 'kf':
                        time_delta = time_ann - this_input_time
                        assert time_delta >= 0
                        new_rst[this_sample_token] = kf_interpolation(
                            kfilter, this_rst, last_rst, this_sample_token, time_delta, this_time_consume, use_velo=use_velo)
                    elif sp_strategy == 'tracker':
                        time_delta = time_ann - this_input_time
                        assert time_delta >= 0
                        new_rst[this_sample_token], count_idx = tracker_interpolation(
                            tracker3d, this_rst, last_rst, this_sample_token, time_delta, this_time_consume, count_idx)

                    else:
                        raise NotImplementedError

                    this_sample_token = input_token_sequence[this_sample_token]
                    if this_sample_token.rsplit('_', 1)[-1] == '':
                        break
                    time_ann += ann_time_itv

                else:
                    # get last rst for calculating velocity
                    last_rst = copy.deepcopy(this_rst)
                    last_input_time = copy.deepcopy(this_input_time)
                    this_rst = next(iter(this_val_preds[match_idx].values()))[
                        'results']
                    this_input_time = next(iter(this_val_preds[match_idx].values()))[
                        'time_input']
                    this_time_consume = this_input_time - last_input_time
                    match_idx += 1
                    if match_idx >= len(this_val_preds):
                        while this_sample_token.rsplit('_', 1)[-1] != '':
                            new_rst[this_sample_token] = modify_token(
                                this_rst, this_sample_token)
                            this_sample_token = input_token_sequence[this_sample_token]
                        break
            tracker3d = None
        return make_annos_correspond(gt_annos_with_frame_id, new_rst)
    
    def split_annos_in_different_distance(self, gt_annos, det_annos, len_patch=5):
        gt_annos_dict, det_annos_dict = {}, {}
        for dist in range(0, 55, len_patch):
            min_dis, max_dis = dist, dist + len_patch
            gt_annos_dict['{}_{}'.format(min_dis, max_dis)] = []
            det_annos_dict['{}_{}'.format(min_dis, max_dis)] = []

            for i, (gt, det) in enumerate(zip(gt_annos, det_annos)):
                gt_boxes = gt['gt_boxes_lidar']
                det_boxes = det['boxes_lidar']
                new_gt = {}
                new_det = {}
                if gt_boxes.shape[0] > 0:
                    dis_norm = np.linalg.norm(gt_boxes[:, :3], axis=1)
                    mask = np.logical_and(dis_norm >= min_dis, dis_norm < max_dis)
                    for key, value in gt.items():
                        if isinstance(value, np.ndarray):
                            new_gt[key] = value[mask]
                        else:
                            new_gt[key] = value
                else:
                    new_gt = gt
                gt_annos_dict['{}_{}'.format(min_dis, max_dis)].append(new_gt)   

                if det_boxes.shape[0] > 0:
                    dis_norm = np.linalg.norm(det_boxes[:, :3], axis=1)
                    mask = np.logical_and(dis_norm >= min_dis, dis_norm < max_dis)
                    for key, value in det.items():
                        if isinstance(value, np.ndarray):
                            new_det[key] = value[mask]
                        else:
                            new_det[key] = value
                else:
                    new_det = det
                det_annos_dict['{}_{}'.format(min_dis, max_dis)].append(new_det)  
        return gt_annos_dict, det_annos_dict            


    def evaluation_offline(self, gt_annos, det_annos, class_names, eval_type):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        if eval_type == '2d':
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result_2d(
                gt_annos, det_annos, class_names)
        elif eval_type == '3d':
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos, det_annos, class_names)
        else:
            raise NotImplementedError
        return ap_result_str, ap_dict

    def evaluation_stream(self, gt_annos, det_annos, class_names, sp_strategy, use_velo=False):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval
        gt_annos, det_annos = self.streaming_simulation(
            gt_annos, det_annos, sp_strategy=sp_strategy, use_velo=use_velo)
        
        # with open('gt_annos_opt.pkl', 'wb') as f:
        #     pickle.dump(gt_annos, f)
        # with open('det_annos_opt.pkl', 'wb') as f:
        #     pickle.dump(det_annos, f)
        # exit()

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos, det_annos, class_names)
        return ap_result_str, ap_dict
    
    def filter_low_score_object(self, det_annos, score_threshold=0.1):
        for det in det_annos:
            mask = det['score'] > score_threshold
            for key, value in det.items():
                if isinstance(value, np.ndarray):
                    det[key] = value[mask]
        return det_annos

    def evaluation(self, det_annos, class_names, eval_metric=['offline'], **kwargs):
        if 'annos' not in self.kitti_infos[0]['infos']['token'].keys():
            return None, {}

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(
            info['infos']['token']['annos']) for info in self.kitti_infos]
        
        if kwargs.get('score_threshold', None) is not None:
            eval_det_annos = self.filter_low_score_object(
                eval_det_annos, score_threshold=kwargs.score_threshold)
        
        use_velo = kwargs.get('use_velo', False)

        if isinstance(eval_metric, str):
            eval_metric = [eval_metric]
        re_result_str = {}
        re_result_dict = {}

        if 'get_distance' in eval_metric:
            eval_gt_annos, eval_det_annos = self.split_annos_in_different_distance(
                eval_gt_annos, eval_det_annos, len_patch=5)
            for key, value in eval_gt_annos.items():
                for metric in eval_metric:
                    eval_type = metric.split('_')[-1]
                    if 'offline' in metric:
                        assert eval_type in ['2d', '3d']
                        re_result_str['{}_{}'.format(metric, key)], re_result_dict['{}_{}'.format(metric, key)] = self.evaluation_offline(
                            value, eval_det_annos[key], class_names, eval_type)
                        continue
                    if 'stream' in metric:
                        assert eval_type in ['kf', 'copy']
                        re_result_str['{}_{}'.format(metric, key)], re_result_dict['{}_{}'.format(metric, key)] = self.evaluation_stream(
                            value, eval_det_annos[key], class_names, eval_type, use_velo=use_velo)
                        continue
                    # raise NotImplementedError(
                    #     f'eval metric of {metric} is not implemented!')

            return re_result_str, re_result_dict

        else:
            for metric in eval_metric:
                eval_type = metric.split('_')[-1]
                if 'offline' in metric:
                    assert eval_type in ['2d', '3d']
                    re_result_str[metric], re_result_dict[metric] = self.evaluation_offline(
                        eval_gt_annos, eval_det_annos, class_names, eval_type)
                    continue
                if 'stream' in metric:
                    assert eval_type in ['kf', 'copy', 'tracker']
                    re_result_str[metric], re_result_dict[metric] = self.evaluation_stream(
                        eval_gt_annos, eval_det_annos, class_names, eval_type, use_velo=use_velo)
                    continue
                # raise NotImplementedError(
                #     f'eval metric of {metric} is not implemented!')

            return re_result_str, re_result_dict


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
        return len(self.kitti_infos)

    def __getitem__(self, index):
        assert self.dataset_cfg.FOV_POINTS_ONLY
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        input_dict = self.load_input_data(info)

        if self.mode == 'train':
            # check gt frame is exist
            check_frame_valid = all([input_dict[self.box3d_supervision]['frame_valid'],
                                     input_dict[self.depth_supervision]['frame_valid']])
            if not check_frame_valid:
                new_index = np.random.randint(0, self.__len__())
                return self.__getitem__(new_index)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
