import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

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
        # batch size
        batch_size = batch_dict['batch_size']
        # (batch, 128, 7)
        rois = batch_dict['rois']
        # 所有的关键点(batch*2048, 4) batch_id, x, y, z
        point_coords = batch_dict['point_coords']
        # 关键点编码的特征(batch*2048, 128)
        point_features = batch_dict['point_features']
        # PKW 前景点对最终的结果预测应该占有更大的权重
        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)
        # global_roi_grid_points (batch*128, 216, 3) local_roi_grid_points(batch*216, 3)
        # (BxN, 6x6x6, 3) global_roi_grid_points是雷达坐标系的grid point；local_roi_grid_points是CCS坐标系下的grid point
        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )
        # (batch*128, 216, 3) --> (batch, 128x6x6x6, 3) == (batch, 27648, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)
        # 关键点的xyz坐标(batch*2048, 3)
        xyz = point_coords[:, 1:4]
        # 保留一个batch中每帧点云的关键点个数
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()

        batch_idx = point_coords[:, 0]

        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()
        # (batch * 27648, 3)
        new_xyz = global_roi_grid_points.view(-1, 3)
        # 存储每一帧点云中所有grid point的数量 [27684, 27684]
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        # pooled_points为所有grid point点在雷达下的坐标(batch*27648, 3)，
        # pooled_features为所有grid point点经过ROI-grid pooling后的特征(batch*27648, 128)
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),  # 关键点的坐标
            xyz_batch_cnt=xyz_batch_cnt,  # 关键点的数量
            new_xyz=new_xyz,  # grid point的坐标
            new_xyz_batch_cnt=new_xyz_batch_cnt,  # grid point的数量
            features=point_features.contiguous(),  # 关键点的特征 (batch * 2, 128)
        )  # (M1 + M2 ..., C)
        # (batch*27648, 3) --> (batch * 128, 6*6*6, 128)
        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        # (batch, 128, 7) --> (batch * 128, 7)
        rois = rois.view(-1, rois.shape[-1])
        # batch * 128
        batch_size_rcnn = rois.shape[0]
        # 每个box中均匀分布的6*6*6 的grid point点的坐标（CCS坐标系） shape : (batch * 128, 216, 3)
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
        # (6, 6, 6)
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = faked_features.nonzero()
        # (B, 6x6x6, 3)
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()
        # (batch * 128,  3)  (l, w, h)
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        # (B, 6x6x6, 3)
        # (dense_idx + 0.5) / grid_size   shape (256, 216, 3)
        # local_roi_size.unsqueeze(dim=1) shape (256, 1, 3)
        # (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1)获取每个grid point点的位置
        #   - (local_roi_size.unsqueeze(dim=1) / 2) 将得到的grid point点转换到box的中心为原点（CCS坐标系）
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # 根据所有的预测结果生成proposal, rois: (B, num_rois, 7+C)
        # roi_scores: (B, num_rois) roi_labels: (B, num_rois)
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        # 训练模式下, 需要对选取的roi进行target assignment，并将ROI对应的GTBox转换到CCS坐标系下
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']  # 将原来的512个proposal替换成128个proposal
                batch_dict['roi_labels'] = targets_dict['roi_labels']  # 将原来的512个proposal的类别替换成128个proposal类别

        # RoI-grid pooling   在每个proposal内部均匀采样6*6*6个grid point,并融合keypoint的特征(batch * 128, 6*6*6, 128)
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        # 6
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch * 128
        batch_size_rcnn = pooled_features.shape[0]
        # (BxN, C, 6, 6, 6) (batch * 128, 6*6*6, 128)-->(batch * 128, 128, 6, 6, 6)
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)
        # (batch * 128, 256, 1)
        """
        Sequential(
          (0): Conv1d(27648, 256, kernel_size=(1,), stride=(1,), bias=False)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
          (3): Dropout(p=0.3, inplace=False)
          (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,), bias=False)
          (5): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): ReLU()
        )
        """
        # pooled_features.view(batch_size_rcnn, -1, 1) shape (batch * 128, 6*6*6*C) C为128，PointNet聚合的特征
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # (B, 1 or 2)     rcnn_cls proposal的置信度  (batch * 128,  1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        # (B, C)      rcnn_reg  proposal的box refinement结果 (batch * 128,  7)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        # 推理模式下，根据微调生成预测结果
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
