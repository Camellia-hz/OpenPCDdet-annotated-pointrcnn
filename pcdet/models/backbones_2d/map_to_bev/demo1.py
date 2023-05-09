import torch
import  torch.nn as nn

class PointPillarScatter(nn.Module):

    def __init__(self, model_fig, grid_size, **kwargs):
        super().__init__()
        self.model_fig = model_fig
        self.grid_size = grid_size
        self.nx, self.ny, self.nz = grid_size
        self.num_bev_features = 64

        def forward(self, batch_dict):
            pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

            batch_spatial_features = []

            batch_size = coords[: 0].max().int().item()+1

            # 创建一个全0矩阵，用来存储当前idx下的点云的所有pillar的特征
            # 64 * [496*432*1] 其中包含有效pillar 和空pillar ，空pillar处不做填充
            for batch_idx in range(batch_size):
                spatial_features = torch.zeros(self.num_bev_features, self.nxself.nz * self.nx * self.ny,
                                               dtype=pillar_features.dtype, device=pillar_features.device)
                # 取出当前点云的数据
                batch_mask = coords[:, 0] == batch_idx
                batch_coords = coords[batch_mask, :]
                # batch_coords: z y x-->0,y,x-->1 2 3  spatial_features-->493 *432 y x
                # 同时求所有点（pillar）的索引
                indices = batch_coords[:, 1] + batch_coords[:, 2] * self.nx + batch_coords[:, 3]
                indices = indices.type(torch.long)

                # 取出索引对应的特征
                pillars = pillar_features[batch_mask, :]
                pillars = pillars.t()

                spatial_features[:, indices] = pillars
                batch_spatial_features.append(spatial_features)

            batch_spatial_features = torch.stack(batch_spatial_features, dim=0)
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features, self.ny,self.nx)
            batch_dict['spatial_features'] =batch_spatial_features
            return batch_dict




