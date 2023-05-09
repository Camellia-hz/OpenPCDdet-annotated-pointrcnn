import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class VoxelRCNNHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        # shape (batch, 128, 7)
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)  # False
        # roi_grid_xyz是雷达坐标系的grid point （Batch * num_of_roi, 6*6*6, 3） ；_是proposal中心坐标点CCS坐标系下的grid point
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)

        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)

        # compute the voxel coordinates of grid points  计算得到每一个grid point在哪一个voxel内，并得到该voxel的坐标
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3) 得到grid point所在voxel的坐标 xyz
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)
        # 创建batch mask
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(
            roi_grid_coords.shape[1])  # 每个batch中所有的proposal有多少个grid point [128*6*6*6, 128*6*6*6]

        """
        这里pool有三个尺度，这里仅仅粘贴出来第一个x_conv2中的pool操作
                ModuleList(
          (0): NeighborVoxelSAModuleMSG(
            (groupers): ModuleList(
              (0): VoxelQueryAndGrouping()
            )
            (mlps_in): ModuleList(
              (0): Sequential(
                (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (mlps_pos): ModuleList(
              (0): Sequential(
                (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (mlps_out): ModuleList(
              (0): Sequential(
                (0): Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
                (1): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): ReLU()
              )
            )
            (relu): ReLU()
          )
        """
        # self.pool_cfg.FEATURES_SOURCE : ['x_conv2', 'x_conv3', 'x_conv4']
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            # 该层3D特征中非空voxel的indices
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                # shape (num_of_voxel, 3) 根据点云的范围，voxel的大小，该3D卷积层的下采样倍数，来获得该层所有非空voxel的中心点坐标
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                # 得到每帧点云在该尺度下有多少个voxel ，cur_voxel_xyz_batch_cnt [num_of_voxel_1, num_of_voxel_2]
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            # get voxel2point tensor,在特征尺度下根据非空的voxel的坐标点，来生成3D密集矩阵数据，
            # 并将非空的voxel坐标点设置为该voxel在一个batch中的索引，空的voxel坐标点数值设置为-1；shape (batch, z, y, x)
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            # 计算得到proposal中每个grid经过3D卷积下采样后的voxel coor坐标
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            # 拼接batch_idx到cur_roi_grid_coords
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()

            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),  # 该3D卷积特征层上非空voxel的坐标 (num_of_voxel, 3)
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,  # 每一帧点云中，在该尺度上voxel的个数
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),  # 每个proposal中roi grid point在点云坐标系下的坐标 单位:米
                new_xyz_batch_cnt=roi_grid_batch_cnt,  # 一批数据中，每帧点云roi grid point的个数
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),  # 该尺度下roi grid point的voxel坐标
                features=cur_sp_tensors.features.contiguous(),  # 该层的3D卷积特征
                voxel2point_indices=v2p_ind_tensor  # 密集3D矩阵
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        # (batch*128, 6x6x6, 32*3)每个grid point拼接来自['x_conv2', 'x_conv3', 'x_conv4']特征
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)

        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])  # (batch*128, 7)
        batch_size_rcnn = rois.shape[0]  # batch*128
        # 每个box中均匀分布的6*6*6 的grid point点的坐标（基于proposal中心的CCS坐标系） shape : (batch * 128, 216, 3)
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        # 将获取的grid point 转换到与之对应roi的旋转角度下       shape : (batch * 128, 216, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        # CCS坐标的XYZ转换到雷达坐标系下
        global_center = rois[:, 0:3].clone()
        # shape : (batch * 128, 216, 3)
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        # global_roi_grid_points是雷达坐标系的grid point；local_roi_grid_points是CCS坐标系下的grid point
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))  # (6, 6, 6)
        dense_idx = faked_features.nonzero()  # (6*6*6, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)
        # (batch * 128,  3)  (l, w, h)
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        # (B, 6x6x6, 3)
        # (dense_idx + 0.5) / grid_size   shape (256, 216, 3)
        # local_roi_size.unsqueeze(dim=1) shape (256, 1, 3)
        # (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1)获取每个grid point点的位置
        #   - (local_roi_size.unsqueeze(dim=1) / 2) 将得到的grid point点转换到box的中心为原点（CCS坐标系）
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # 根据所有的预测结果生成proposal, rois: (B, num_rois, 7+C)
        # roi_scores: (B, num_rois) roi_labels: (B, num_rois)
        # 训练生成512个ROI，推理生成100个ROI
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        # 训练模式下, 需要对选取的roi进行target assignment，并将ROI对应的GTBox转换到CCS坐标系下
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling    pooled_features shape (batch * 128, 6x6x6, 96)
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement (batch * 128, 6x6x6, 96) --> (batch, 128, 6x6x6x96)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        """
        使用一个两层的MLP生成最终的特征
        Sequential(
        (0): Linear(in_features=20736, out_features=256, bias=False)
        (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Dropout(p=0.3, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=False)
        (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): ReLU(inplace=True)
        )
        """
        shared_features = self.shared_fc_layer(pooled_features)
        # (256, 1)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        # (256, 7)
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
