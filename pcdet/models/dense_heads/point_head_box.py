import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate


class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """

    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)

        self.predict_boxes_when_training = predict_boxes_when_training
        # 创建cls预测模块
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        # 初始化anchor和GT的编码器函数
        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        # 创建box预测模块
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
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
        # 得到每个点的坐标 shape（bacth * 16384, 4），其中4个维度分别是batch_id，x，y，z
        point_coords = input_dict['point_coords']
        # 取出gt_box，shape （batch， num_of_GTs, 8），
        # 其中维度8表示 x, y, z, l, w, h, heading, class_id
        gt_boxes = input_dict['gt_boxes']
        # 确保维度正确
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        # 在训练的过程中，需要忽略掉点云中离GTBox较近的点，因为3D的GTBox也会有扰动，
        # 所以这里通过将每一个GT_box都在x，y，z方向上扩大0.2米，
        # 来检测哪些点是属于扩大后才有的点，增强点云分割的健壮性
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        """
        assign_stack_targets函数完成了一批数据中所有点的前背景分配，
        并为每个前景点分配了对应的类别和box的7个回归参数，xyzlwhθ
        """
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        # 计算第一阶段box的回归损失和分类损失的和
        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
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
        # False
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']  # 从字典中每个点的特征 shape （batch * 16384, 128）
        """ 点分类的网络详情
        Sequential(
            (0): Linear(in_features=128, out_features=256, bias=False)
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Linear(in_features=256, out_features=256, bias=False)
            (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            (6): Linear(in_features=256, out_features=3, bias=True)
        )
        """
        # 对每个点进行分类 (batch * 16384, num_class)
        point_cls_preds = self.cls_layers(point_features)
        """ 点生成proposal的网络详情，其中这里作者使用了residual-cos-based来编码θ，也就是角度被 (cos(∆θ), sin(∆θ))来编码，所以最终回归的参数是8个
        Sequential(
            (0): Linear(in_features=128, out_features=256, bias=False)
            (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Linear(in_features=256, out_features=256, bias=False)
            (4): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (5): ReLU()
            (6): Linear(in_features=256, out_features=8, bias=True)
        )
        """
        # 点回归proposal (batch * 16384, box_code_size)
        point_box_preds = self.box_layers(point_features)
        # 从每个点的分类预测结果中，取出类别预测概率最大的结果  (batch * 16384, num_class) --> (batch * 16384, )
        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        # 将类别预测分数经过sigmoid激活后放入字典中
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)
        # 将点的类别预测结果和回归结果放入字典中
        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        # 如果在训练模式下，需要根据GTBox来生成对应的前背景点，用于点云的前背景分割，给后面计算前背景分类loss
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            # 将一个batch中所有点的GT类别结果放入字典中 shape (batch * 16384)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            # 将一个batch中所有点的GT_box编码结果放入字典中 shape (batch * 16384) shape (batch * 16384, 8)
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
        # 训练和预测都需要生成第一阶段的proposal 然后给第二阶段来refine
        if not self.training or self.predict_boxes_when_training:
            # 生成预测的box
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                # 所有点的xyz坐标（batch*16384, 3）
                points=batch_dict['point_coords'][:, 1:4],
                # 所有点的预测类别（batch*16384, 3）
                point_cls_preds=point_cls_preds,
                # 所有点的box预测（batch*16384, 8）
                point_box_preds=point_box_preds
            )
            # 所有点的类别预测结果 (batch * 16384, 3)
            batch_dict['batch_cls_preds'] = point_cls_preds
            # 所有点的回归预测结果 (batch * 16384, 7)
            batch_dict['batch_box_preds'] = point_box_preds
            # 所有点的在batch中的索引 (batch * 16384, )
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            # loss计算中，是否需要对类别预测结果进行normalized
            batch_dict['cls_preds_normalized'] = False
        # 第一阶段生成的预测结果放入前向传播字典
        self.forward_ret_dict = ret_dict

        return batch_dict
