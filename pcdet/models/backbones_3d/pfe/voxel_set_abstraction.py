import math
import numpy as np
import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


# 完成双线性插值
def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi

    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

    return sampled_points, point_mask


def sector_fps(points, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list = []
    xyz_batch_cnt = []
    num_sampled_points_list = []

    for k in range(num_sectors):
        mask = (sector_idx == k)
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(
                min(cur_num_points, math.ceil(ratio * num_sampled_points))
            )

    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        print(f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}')

    xyz = torch.cat(xyz_points_list, dim=0)

    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()

    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
        xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
    ).long()

    sampled_points = xyz[sampled_pt_idxs]

    return sampled_points


class VoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += cur_num_c_out

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points']
            )

            c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    # 获取从BEV视角通过插值得到的关键点特征
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        # 得到该关键点对应的voxel的x坐标 shape : (2048*batch,)
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        # 得到该关键点对应的voxel的y坐标 shape : (2048*batch,)
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        # x坐标除下采样倍数 shape : (2048*batch,)
        x_idxs = x_idxs / bev_stride
        # y坐标除下采样倍数 shape : (2048*batch,)
        y_idxs = y_idxs / bev_stride
        # 结果列表
        point_bev_features_list = []
        for k in range(batch_size):  # 逐帧进行插值操作
            bs_mask = (keypoints[:, 0] == k)  # 当前帧点云的mask

            cur_x_idxs = x_idxs[bs_mask]  # 取出属于当前帧关键点的x坐标
            cur_y_idxs = y_idxs[bs_mask]  # 取出属于当前帧关键点的y坐标
            # 对当前帧的BEV特征图进行维度转换 (C, 200, 176)  --> (200, 176, C)
            cur_bev_features = bev_features[k].permute(1, 2, 0)
            # 通过双线性插值获得关键点的特征  shape (2048, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            # 结果放入列表中
            point_bev_features_list.append(point_bev_features)
        # 将通过插值得到的关键点特征在第0维度进行拼接 （2048*batch, C）
        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """

        sampled_points, _ = sample_points_with_roi(

            rois=roi_boxes, points=points,

            sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,

            num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000)
        )

        sampled_points = sector_fps(

            points=sampled_points, num_sampled_points=self.model_cfg.NUM_KEYPOINTS,

            num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS
        )
        return sampled_points

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            # 取出所有原始点的xyz数据, shape (N_raw_point, 3)   batch_id, x, y, z, r
            src_points = batch_dict['points'][:, 1:4]
            # 取出所有点对应的batch_id, shape (N_raw_point,)
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        # 创建用于存储关键点的列表
        keypoints_list = []
        # 逐帧点云获取关键点
        for bs_idx in range(batch_size):
            # 得到属于当前帧的mask
            bs_mask = (batch_indices == bs_idx)
            # 索引出所有属于当前帧的点 shape (1, num_of_point, 3)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            # 根据采样的模式，此处是最远点采样，
            # 在KITTI中关键点个数为2048，在Waymo中关键点个数为4096
            # cur_pt_idxs shape (1, 2048) 为关键点在原始点云中的索引
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()
                # 如果采样点数小于设置的采样点数，则重复采样至设置采样点数
                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]
                # 根据cur_pt_idxs取出当前帧的所有关键点 shape (1, 2048, 3)
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)


            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                cur_keypoints = self.sectorized_proposal_centric_sampling(
                    roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0]
                )
                bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
                keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            else:
                raise NotImplementedError
            # 将当前帧中选取的关键点加入keypoints_list列表
            keypoints_list.append(keypoints)
        # 在第0维度将所有关键点进行拼接 （2048*batch_size, 3）
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        # 将keypoints数据中该点所属帧 (batch_idx, x, y, z)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size,
                                     device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
        return keypoints

    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func: PointNet++的SA操作
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        # 用于存储一帧中，原始点云的总数
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            # 用于计算一个batch中每帧的原始点云的个数 shape （N0, N1, N2, N3）
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()
        # pooled_points : (2048*batch, 3)  pooled_features : (2048*batch, 32)
        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        """
        ===============================
        通过最远点采样，获取每一帧点云的关键点
        ===============================
        """
        keypoints = self.get_sampled_points(batch_dict)

        """
        ============================
        Extended VSA中对BEV插值SA操作
        ============================
        """
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            # 通过对BEV的特征进行插值获取关键点的特征数据
            point_bev_features = self.interpolate_from_bev_features(
                # 所有的关键点
                keypoints,
                # 将高度进行堆叠后的BEV特征图
                batch_dict['spatial_features'],
                # batch size
                batch_dict['batch_size'],
                # 高度堆叠后的BEV特征图的下采样倍数，默认是8倍
                bev_stride=batch_dict['spatial_features_stride']
            )
            # 将插值得到的关键点特征存入列表
            point_features_list.append(point_bev_features)
        # batch_size
        batch_size = batch_dict['batch_size']
        # 取出所有关键点的xyz坐标
        new_xyz = keypoints[:, 1:4].contiguous()
        # 存储每帧点云关键点的个数
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        # 获取batch中每帧点云中关键点的个数
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()

        """
        所有SA层的配置中MLPS为1*1的卷积，DOWNSAMPLE_FACTOR该层对应的下采样倍数
        
        """

        """
        ============================
        Extended VSA中对原始点云的操作
        ============================
        POOL_RADIUS为该层对应的采样半径，
        NSAMPLE为半径内最大的采样点数
        SA_LAYER:
            raw_points:
                MLPS: [[16, 16], [16, 16]]
                POOL_RADIUS: [0.4, 0.8]
                NSAMPLE: [16, 16]
        ============================
        """
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            # 得到所有一批数据中所有原始点云数据 shape （N, 5） batch_id, x, y, z, r
            raw_points = batch_dict['points']
            # pooled_features: (2048 * batch, 32)
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                # batch size
                batch_size=batch_size,
                # 聚合函数，这里采用PointNet++的Set Abstraction
                aggregate_func=self.SA_rawpoints,
                # 原始点的xyz数据
                xyz=raw_points[:, 1:4],
                # 每个原始点云的 reflect intensity
                xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
                # 每个点所在batch中的索引
                xyz_bs_idxs=raw_points[:, 0],
                # 关键点的xyz shape : (2048, 3)
                new_xyz=new_xyz,
                # 一批数据中关键点的个数
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                # False
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False),
                # None
                radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                # None
                rois=batch_dict.get('rois', None)
            )

            point_features_list.append(pooled_features)

        """
        ====================================
        VSA中对不同尺度3D CNN的voxel-wise的操作
        1x, 2x, 4x, 8x
        ====================================
        POOL_RADIUS为该层对应的采样半径，
        NSAMPLE为半径内最大的采样点数
        x_conv1:
                DOWNSAMPLE_FACTOR: 1
                MLPS: [[16, 16], [16, 16]]
                POOL_RADIUS: [0.4, 0.8]
                NSAMPLE: [16, 16]
        x_conv2:
            DOWNSAMPLE_FACTOR: 2
            MLPS: [[32, 32], [32, 32]]
            POOL_RADIUS: [0.8, 1.2]
            NSAMPLE: [16, 32]
        x_conv3:
            DOWNSAMPLE_FACTOR: 4
            MLPS: [[64, 64], [64, 64]]
            POOL_RADIUS: [1.2, 2.4]
            NSAMPLE: [16, 32]
        x_conv4:
            DOWNSAMPLE_FACTOR: 8
            MLPS: [[64, 64], [64, 64]]
            POOL_RADIUS: [2.4, 4.8]
            NSAMPLE: [16, 32]
        =====================================
        """

        for k, src_name in enumerate(self.SA_layer_names):
            # 获取非空voxel的xyz的index shape : (16000*batch, 4)
            # 16000为训练中选取的最大非空voxel数
            # 4-->(batch_idx, z, y, x)
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            # 获取非空voexel的特征
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
            # 获取每个voxel在点云坐标系中xyz的中心点坐标 (16000*batch, 3)
            xyz = common_utils.get_voxel_centers(

                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )
            # VSA操作
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                # batch_size
                batch_size=batch_size,
                # 第K层上的SA网络
                aggregate_func=self.SA_layers[k],
                # 所有voxel在点云坐标系下中心点的坐标
                xyz=xyz.contiguous(),
                # 所有voxel在第K层的特征
                xyz_features=cur_features,
                # 所有点在batch中的mask
                xyz_bs_idxs=cur_coords[:, 0],
                # 关键点坐标 (2048, 3)
                new_xyz=new_xyz,
                # 关键点个数
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                # False
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                # None
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                # None
                rois=batch_dict.get('rois', None)
            )
            # 得到的pooled_features 维度为 (2048 * batch, C)
            point_features_list.append(pooled_features)

        """
        [
        (2048 * batch, 256) BEV视角下点特征数据
        (2048 * batch, 32)  原始点云下特征数据
        (2048 * batch, 32)  x_conv1 第一次稀疏卷积后特征数据
        (2048 * batch, 64)  x_conv2 第二次稀疏卷积后特征数据
        (2048 * batch, 128) x_conv3 第三次稀疏卷积后特征数据
        (2048 * batch, 128) x_conv4 第四次稀疏卷积后特征数据
        ]
        """
        # point_features (2048 * batch, 640)
        point_features = torch.cat(point_features_list, dim=-1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        """
        将不同尺度拼接的特征使用一个简单的MLP网络对处在不同尺度的特征进行聚合
        vsa_point_feature_fusion
        Sequential(
          (0): Linear(in_features=640, out_features=128, bias=False)
          (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        """
        # point_features (2048 * batch, 640)-->(2048 * batch, 128)
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        # (batch*2048, C)
        batch_dict['point_features'] = point_features
        # (batch*2048, 4)    4-->(batch_id, x, y, z)
        batch_dict['point_coords'] = keypoints
        return batch_dict
