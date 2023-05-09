import numpy as np
import torch
import torch.nn as nn


class BaseBEVbackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        """
                   BACKBONE_2D:
                   NAME: BaseBEVBackbone
                   LAYER_NUMS: [3, 5, 5]
                   LAYER_STRIDES: [2, 2, 2]
                   NUM_FILTERS: [64, 128, 256]
                   UPSAMPLE_STRIDES: [1, 2, 4]
                   NUM_UPSAMPLE_FILTERS: [128, 128, 128]
        """
        super().__init__()
        self.model_cfg = model_cfg
        # 读取下采样的所有参数
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        # 读取上采样的所有参数
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            num_upsample_filters = upsample_strides = []
        # 三块，每块layersnum
        num_levels = len(num_filters)
        # 下采样的输入尺寸是原始数据输入+num_filters中前n-1个  最后一个作为输出
        # 每块的输出尺寸为num_filters
        c_in_list = [input_channels, *num_filters[:-1]]

        # 定义下采样和上采样的网络块
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        # 定义每一块
        for idx in range(num_levels):
            # 单独定义第一层具有尺寸变化的层
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3,
                          stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            # 再定义后面的尺寸不变层 LAYER_NUMS: [3, 5, 5]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3,
                               padding=1, bias=False),
                     nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                     nn.ReLU()]
                )

            self.blocks.append(nn.Sequential(*cur_layers))
            # 定义上采样层 ，每一块下采样都对应一个上采样层，该层接收下采样的结果
            if len(upsample_strides) > 0:  # 构造上采样层  # (1, 2, 4)
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx],
                                           kernel_size=num_upsample_filters[idx],
                                           stride=num_upsample_filters[idx], padding=0, bias=False
                                           ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        # 将上采样结果 按照通道维度进行拼接
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1],
                                       padding=0, bias=False
                                       ),
                    nn.BatchNorm2d(c_in),
                    nn.ReLU()
                ))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        #  需要伪图像数据

        spatial_features = data_dict['spatial_features']
        # 保存每个上采样结果，方便拼接
        ups = []
        ret_dict = []
        # 输入维度：[batch_size, 64, 496, 432]
        x = spatial_features

        for i in range(self.blocks):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            # 将不同块的下采样结果保存在字典中
            ret_dict['spatial_features_%dx' % stride] = x

            # 对当前下采样块立刻进行上采样

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        # 上采样结果拼接
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict