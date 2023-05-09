import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class PointRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        use_bn = self.model_cfg.USE_BN
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        self.num_prefix_channels = 3 + 2  # xyz + point_scores + point_depth

        xyz_mlps = [self.num_prefix_channels] + self.model_cfg.XYZ_UP_LAYER

        shared_mlps = []

        for k in range(len(xyz_mlps) - 1):

            shared_mlps.append(nn.Conv2d(xyz_mlps[k], xyz_mlps[k + 1], kernel_size=1, bias=not use_bn))

            if use_bn:
                shared_mlps.append(nn.BatchNorm2d(xyz_mlps[k + 1]))
            shared_mlps.append(nn.ReLU())

        self.xyz_up_layer = nn.Sequential(*shared_mlps)

        c_out = self.model_cfg.XYZ_UP_LAYER[-1]

        self.merge_down_layer = nn.Sequential(
            nn.Conv2d(c_out * 2, c_out, kernel_size=1, bias=not use_bn),
            *[nn.BatchNorm2d(c_out), nn.ReLU()] if use_bn else [nn.ReLU()]
        )

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + self.model_cfg.SA_CONFIG.MLPS[k]

            npoint = self.model_cfg.SA_CONFIG.NPOINTS[k] if self.model_cfg.SA_CONFIG.NPOINTS[k] != -1 else None

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModule(
                    npoint=npoint,
                    radius=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsample=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=True,
                    bn=use_bn
                )
            )
            channel_in = mlps[-1]

        self.cls_layers = self.make_fc_layers(
            input_channels=channel_in, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=channel_in,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        self.roipoint_pool3d_layer = roipoint_pool3d_utils.RoIPointPool3d(
            num_sampled_points=self.model_cfg.ROI_POINT_POOL.NUM_SAMPLED_POINTS,
            pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
        )
        self.init_weights(weight_init='xavier')

    # 初始化网络的参数
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

    def roipool3d_gpu(self, batch_dict):
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
        # batch_size
        batch_size = batch_dict['batch_size']
        # 得到每个点云属于batch中哪一帧的mask (batch*2， )
        batch_idx = batch_dict['point_coords'][:, 0]
        # 得到每个点云在激光雷达中的坐标 (batch * 16384, 3)
        point_coords = batch_dict['point_coords'][:, 1:4]
        # 得到每个点在第一个子网络的特征数据（PointNet++）输出的结果 (batch * 16384, 128)
        point_features = batch_dict['point_features']
        # 得到生成的rois， （batch， num_rois, 7） num_rois训练为128，测试为100
        rois = batch_dict['rois']  # (B, num_rois, 7 + C)
        # 该参数用来存储一个batch中每帧点云中有多少个点，并用于匹配一批数据中每帧点的数量是否一致
        batch_cnt = point_coords.new_zeros(batch_size).int()
        # 确保一批数据中点的数量一样
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert batch_cnt.min() == batch_cnt.max()
        # 得到每个点类别预测分数 (batch*2， )，并从计算图中拿出来
        point_scores = batch_dict['point_cls_scores'].detach()
        # 计算雷达坐标系下每个点到原点的距离，这里采用了L2范数（欧氏距离）来计算 (batch*2， )
        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_POOL.DEPTH_NORMALIZER - 0.5
        # 点云中每个点在此处融合了，点的预测分数，每个点的深度信息，和每个点经过PointNet++得到的特征数据
        point_features_list = [point_scores[:, None], point_depths[:, None], point_features]
        # shape （batch * 16384, 130）将所有的这个特征在最后一个维度拼接在一起
        point_features_all = torch.cat(point_features_list, dim=1)
        # 每个点在点云中的的坐标   shape （batch * 16384, 3） --> （batch , 16384, 3）
        batch_points = point_coords.view(batch_size, -1, 3)
        # shape （batch * 16384, 130）--> （batch , 16384, 130）
        batch_point_features = point_features_all.view(batch_size, -1, point_features_all.shape[-1])
        # 不保存计算图  根据生成的proposal来池化每个roi内的特征
        with torch.no_grad():
            # pooled_features为(batch, num_of_roi, num_sample_points, 3 + C)
            # pooled_empty_flag:(B, num_rois)反映哪些proposal中没有点在其中
            pooled_features, pooled_empty_flag = self.roipoint_pool3d_layer(
                batch_points, batch_point_features, rois
            )

            # canonical transformation (batch, num_of_roi, 3)
            roi_center = rois[:, :, 0:3]
            # 池化后的proposal转换到以自身ROI中心为坐标系
            # （batch, num_rois, num_sampled_points, 3 + C = 133）
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            # （batch, num_rois, num_sampled_points, 133） --> （batch * num_rois, num_sampled_points, 133）
            pooled_features = pooled_features.view(-1, pooled_features.shape[-2], pooled_features.shape[-1])
            # 上面完成平移操作后，下面完成旋转操作，使x方向朝向车头方向，y垂直于x，z向上
            # （openpcdet中，x向前，y向左，z向上，x到y逆时针为正）
            pooled_features[:, :, 0:3] = common_utils.rotate_points_along_z(
                pooled_features[:, :, 0:3], -rois.view(-1, rois.shape[-1])[:, 6]
            )
            # 将proposal池化后没有点在内的proposal置0
            pooled_features[pooled_empty_flag.view(-1) > 0] = 0

        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:

        """
        # 生成proposal；在训练时，NMS保留512个结果，NMS_thresh为0.8；在测试时，NMS保留100个结果，NMS_thresh为0.85
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        # 在训练模式时，需要为每个生成的proposal匹配到与之对应的GT_box
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # (bacth * num_of_roi, num_sampled_points, 133) num_sampled_points:512
        pooled_features = self.roipool3d_gpu(batch_dict)  # (total_rois, num_sampled_points, 3 + C)
        # (bacth * num_of_roi, 5, num_sampled_points, 1)
        xyz_input = pooled_features[..., 0:self.num_prefix_channels].transpose(1, 2).unsqueeze(dim=3).contiguous()
        # (bacth * num_of_roi, 128, num_sampled_points, 1) 完成canonical transformation后的mlp操作
        xyz_features = self.xyz_up_layer(xyz_input)
        # (bacth * num_of_roi, 128, num_sampled_points, 1)
        point_features = pooled_features[..., self.num_prefix_channels:].transpose(1, 2).unsqueeze(dim=3)
        # (bacth * num_of_roi, 256, num_sampled_points, 1) 将池化特征和点的特征进行拼接（Merged Features）
        merged_features = torch.cat((xyz_features, point_features), dim=1)
        # (bacth * num_of_roi, 128, num_sampled_points, 1) # 将拼接的特征放回输入之前的大小  channel : 256->128
        merged_features = self.merge_down_layer(merged_features)
        # 同之前的SA操作 进入Point Cloud Encoder
        l_xyz, l_features = [pooled_features[..., 0:3].contiguous()], [merged_features.squeeze(dim=3).contiguous()]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        # (total_rois, num_features, 1)
        shared_features = l_features[-1]
        # (total_rois, num_features, 1) --> (total_rois, 1)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # (total_rois, num_features, 1) --> (total_rois, 7)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

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
