import torch

from ...utils import box_utils
from .point_head_template import PointHeadTemplate


class PointHeadSimple(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """

    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        # (batch * 2048, 4)  4代表 batch_id, x, y, z
        point_coords = input_dict['point_coords']
        # (batch, num_of_gt, 8)得到GTBox  8代表 (x,y,z,l,w,h,θ,class)
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        # 每个GTBox的长宽高扩大0.2米，如果关键点在这扩大的范围内，则标签置为-1，不计算前背景分割loss
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        # 这里取出vsa_point_feature_fusion之前的特征 shape : (batch * 2048, 640)
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']

        """
        前背景分类的MLP设置
            Sequential(
            (0): Linear(in_features=640, out_features=256, bias=False)
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Linear(in_features=256, out_features=256, bias=False)
            (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            (6): Linear(in_features=256, out_features=1, bias=True)
            )
        """
        # point_cls_preds  shape : (batch * 2048, 1)
        point_cls_preds = self.cls_layers(point_features)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }
        # 将预测结果用sigmoid函数映射到0-1之间，得到前背景分类概率
        # PKW模块的权重调整会在PVRCNNHead模块中进行,将预测结果放入batch_dict
        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        # 训练模型下，需要对关键点预测进行target assignment, 前景为1, 背景为0
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            # 存储所有关键点属于前背景的mask
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict
        return batch_dict
