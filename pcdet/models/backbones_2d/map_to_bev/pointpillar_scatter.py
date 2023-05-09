import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
       对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """

    def __init__(self, model_cfg, grid_size, **  kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES  # 64
        self.nx, self.ny, self.nz = grid_size  # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
       Args:
           pillar_features:(M,64)
           coords:(M, 4) 第一维是batch_index 其余维度为xyz, 表示每个pillar的zyx坐标
                    这里的坐标应该是相对坐标，比如一个pillar的坐标是(0, 200, 200)
       Returns:
           batch_spatial_features:(batch_size, 64, 496, 432) B*C*H*W--> 伪图像形式的尺寸
       """
        # 拿到经过前面pointnet处理过后的pillar数据和每个pillar所在点云中的坐标位置
        # pillar_features 维度 （M， 64）
        # coords 维度 （M， 4）第0维是在dataset.py中对每个点云加入的一致索引
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        # 将转换成为伪图像的数据存在到该列表中
        batch_spatial_features = []
        # 根据batch_index，获取batch_size大小, coords[:, 0] 第0维是index
        batch_size = coords[:, 0].max().int().item() + 1

        # batch中的每个数据独立处理
        for batch_idx in range(batch_size):
            # 创建一个空间坐标所有用来接受pillar中的数据
            # self.num_bev_features是64
            # self.nz * self.nx * self.ny是生成的空间坐标索引 [496, 432, 1]的乘积
            # spatial_feature 维度 (64,214272)
            # 其中包含有效pillar 和空pillar ，空pillar处不做填充
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)  # (64,214272)-->1x432x496=214272

            # 从coords[:, 0]取出该batch_idx的数据mask 返回mask，[True, False...]
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标，提取整个batch中一个点云的所有pillar所在的坐标
            this_coords = coords[batch_mask, :]
            # this_coords中存储的坐标是z,y和x的形式,且只有一层，因此计算索引的方式如下
            # 平铺后需要计算前面有多少个pillar 一直到当前pillar的索引
            """
            因为前面是将所有数据flatten成一维的了，相当于一个图片宽高为[496, 432]的图片 
            被flatten成一维的图片数据了，变成了496*432=214272;
            而this_coords中存储的是平面（不需要考虑Z轴）中一个点的信息，所以要
            将这个点的位置放回被flatten的一位数据时，需要计算在该点之前所有行的点总和加上
            该点所在的列即可
            """
            # 这里得到所有非空pillar在伪图像的对应索引位置，实际上，coordinate保存的pillar的坐标
            # 已经是相对坐标了，为什么还要返回到伪图像中的坐标呢？因为有的pillar是无效的，空的，不含有特征，
            # 因此，我们需要返回有效pillar的坐标，并将pillar的特征插入到该位置处
            # this_coords[:, 1--3] 分别表示zyx坐标，0是idx，z=0
            # 加入假如某个pillar坐标为(0,210,200) 分别为zyx
            # 那么它在214272的索引是， 210*432 +200(前面有210行*432个pillar+ 当前行的索引)
            # 最后indices维度: M*1 M表示当前点云的pillar数
            # 同时求所有点（pillar）的索引
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features, pillars: M*64(因为有batch_mask，所有仅有batc_idx才有效)
            pillars = pillar_features[batch_mask, :]
            # 将Tensor进行转置
            pillars = pillars.t()
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, 214272)
            batch_spatial_features.append(spatial_feature)

        # 在第0个维度(batch维度)将所有的数据堆叠在一起，和cat不同的是，这里会单独增加一个维度用来堆叠
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # reshape回原空间(伪图像)    (4, 64, 214272)--> (4, 64, 496, 432) -->B, C, H, W
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        # 将结果加入batch_dict
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
