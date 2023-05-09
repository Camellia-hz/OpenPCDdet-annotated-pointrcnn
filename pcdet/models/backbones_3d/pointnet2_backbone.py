import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]

        # 初始化PointNet++中的SetAbstraction
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        # 初始化PointNet++中的feature back-probagation
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    # 处理点云数据，用于得到（batch，n_points,xyz）的形式
    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]  # 经过预处理后，每个点云的第一个维度数据存放的是该点云在batch中的索引
        xyz = pc[:, 1:4].contiguous()  # 得到所有的点云数据
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)  # 得到除了xyz中的其他数据，比如intensity
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        # batch_size
        batch_size = batch_dict['batch_size']
        # (batch_size * 16384, 5)
        points = batch_dict['points']
        # 得到每个点云的batch索引，和所有的点云的xyz、intensity数据
        batch_idx, xyz, features = self.break_up_pc(points)
        # 创建一个全0的xyz_batch_cnt，用于存放每个batch中总点云个数是多少
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        # 在训练的过程中，一个batch中的所有的点云的中点数量需要相等，
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        # shape ：（batch_size， 16384, 3）
        xyz = xyz.view(batch_size, -1, 3)
        # shape : (batch_size, 1, 16384)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() \
            if features is not None else None
        # 定义一个用来存放经过PointNet++的数据结果，用于后面PointNet++的feature back-probagation层
        l_xyz, l_features = [xyz], [features]
        # 使用PointNet++中的SetAbstraction模块来提取点云的数据
        for i in range(len(self.SA_modules)):
            """
            最远点采样的点数
            NPOINTS: [4096, 1024, 256, 64]
            # BallQuery的半径
            RADIUS: [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
            # BallQuery内半径内最大采样点数（MSG）
            NSAMPLE: [[16, 32], [16, 32], [16, 32], [16, 32]]
            # MLPS的维度变换
            # 其中[16, 16, 32]表示第一个半径和采样点下的维度变换，
            [32, 32, 64]表示第二个半径和采样点下的维度变换，以下依次类推
            MLPS: [[[16, 16, 32], [32, 32, 64]],
                   [[64, 64, 128], [64, 96, 128]],
                   [[128, 196, 256], [128, 196, 256]],
                   [[256, 256, 512], [256, 384, 512]]]
            """
            """
            param analyze:
                li_xyz shape:(batch, sample_n_points, xyz_of_centroid_of_x)
                li_features shape:(batch, channel, Set_Abstraction)
            detail:
            1、li_xyz shape:(batch, 4096, 3), li_features shape:(batch, 32+64, 4096)
            2、li_xyz shape:(batch, 1024, 3), li_features shape:(batch, 128+128, 1024)
            3、li_xyz shape:(batch, 256, 3), li_features shape:(batch, 256+256, 256)
            4、li_xyz shape:(batch, 64, 3), li_features shape:(batch, 512+512, 64)
            """
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])

            l_xyz.append(li_xyz)

            l_features.append(li_features)
        # PointNet++中的feature back-probagation层,从已经知道的特征中，通过距离插值的方式来计算出上一个点集中未知点的特征信息
        """
        其中：
        i=[-1, -2, -3, -4]
        以-1为例：
        unknown的特征是上一层点集中所有的点的坐标，known是当前已知特征的点的坐标，feature是对应的特征
        在已经知道的点中，找出三个与之最近的三个不知道的点，然后对这三个点根据距离计算插值，
        得到插值结果（bacth， 1024, 256），再将插值得到的特征和上一层计算的特征在维度上进行
        拼接的得到（bacth， 1024+512, 256），并进行一个mlp（代码实现中用的1*1的卷积完成）操作
        来进行降维得到上一层的点的特征，（bacth，512, 256）。这样就将深层的信息，传递回去了。
        """
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            unknown = l_xyz[i - 1]
            known = l_xyz[i]
            unknow_feats = l_features[i - 1]
            known_feats = l_features[i]
            res = self.FP_modules[i](
                unknown, known, unknow_feats, known_feats
            )  # (B, C, N)
            l_features[i - 1] = res
            # l_features[i - 1] = self.FP_modules[i](
            #     l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            # )  # (B, C, N)

        """
        经过PointNet++的feature back-probagation处理后，
        l_feature的结果是
        [
            [batch, 128, 16384],
            [batch, 256, 4096],
            [batch, 512, 1024],
            [batch, 512, 256],
            [batch, 1024, 64],
        ]
        """
        # 将反向回传得到的原始点云数据进行维度变换 （batch, 128, 16384）--> (batch, 16384, 128)
        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        # 得到的结果存放入batch_dict  (batch, 16384, 128) --> (batch * 16384, 128)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        # (batch * 16384, 1)  (batch * 16384, 3)  变回输入的形式
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)

        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)

        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict
