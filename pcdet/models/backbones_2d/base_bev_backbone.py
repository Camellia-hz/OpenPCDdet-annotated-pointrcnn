import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):

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
        # 读取下采样层参数
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS  # [3, 5, 5]
            layer_strides = self.model_cfg.LAYER_STRIDES  # [2, 2, 2]
            num_filters = self.model_cfg.NUM_FILTERS  # [64, 128, 256]
        else:
            layer_nums = layer_strides = num_filters = []
        # 读取上采样层参数
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS  # [128, 128, 128]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES  # [1, 2, 4]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)  # 3层
        # [64, 64, 128]，(*表示取值，在列表中使用，num_filters取除了最后一个元素)
        # input_channels: 64, num_filters[:-1]：64, 128
        # 下采样有三个块，第一块第一层内的输出通道与原始数据的输入通道一致，
        # 之后不在变化，只有在每一块的第一层接收上一块的输入时，通道数翻倍，尺寸 H W 减半
        # 下采样的输入尺寸是原始数据输入+num_filters中前n-1个  最后一个作为输出
        # 每块的输出尺寸为num_filters
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):  # (64,64)-->(64,128)-->(128,256) # 这里为cur_layers的第一层且stride=2
            # 单独把每一块的第一层拿出来，因为有通道和尺寸上的变化
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    # c_in_list: [64, 64, 128], num_filters: 64, 128, 256
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    # layer_strides：[2, 2, 2]
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            # [3, 5, 5], 这样每一块再加上第一层--> 4, 6, 6
            for k in range(layer_nums[idx]):  # 根据layer_nums堆叠卷积层
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            # 在block中添加该层
            # *作用是：将列表解开成几个独立的参数，传入函数 # 类似的运算符还有两个星号(**)，是将字典解开成独立的元素作为形参
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:  # 构造上采样层  # (1, 2, 4)
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        # 针对每一块的输出通道作为转置卷积的输入通道，一共三块下采样层，对应三块上采样
                        # 上采样到每一块的输出为2C,H/2,W/2
                        # 第二块：out:2C, H/4, W/4-->deconvd: k=2,s=2  in_size = k-2p+(out-1)*s
                        # 代入，输出为 2C,H/2,W/2
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],  # [128, 128, 128]
                            upsample_strides[idx],  # kernel_size = [1, 2, 4]
                            # 转置卷积: 还原in_size = k-2p+(out-1)*s
                            stride=upsample_strides[idx], bias=False  # stride = [1, 2, 4]
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
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

        c_in = sum(num_upsample_filters)  # 384
        # 下面不执行：3 > 3 False, 效果是 当上采样的层数多余块数时，我们将上采样最后合并到的块再进行上采样，
        # 但是通道数不变，尺寸也不变
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        # 输出特征384
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features : (batch, c, W, H) 仅使用伪图像特征
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        # 保存每个上采样结果，方便拼接
        ups = []
        ret_dict = {}
        # 输入维度：[batch_size, 64, 496, 432]
        x = spatial_features

        # 对不同的分支部分分别进行conv和deconv的操作
        for i in range(len(self.blocks)):

            # 下采样
            # 下采样之后，x的shape分别为：
            # torch.Size([batch_size, 64, 248, 216])，
            # torch.Size([batch_size, 128, 124, 108])，
            # torch.Size([batch_size, 256, 62, 54])

            # torch.Size([batch_size, 128, 248, 216])，
            # torch.Size([batch_size, 128, 248, 216])，
            # torch.Size([batch_size, 128, 248, 216])，
            # torch.Size([batch_size, 384, 248, 216])，
            x = self.blocks[i](x)
            # 三次分别为2，4，8 就是每一块的输出对原始inputs尺寸减小了多少倍
            stride = int(spatial_features.shape[2] / x.shape[2])
            # 将不同块的下采样结果保存在字典中
            ret_dict['spatial_features_%dx' % stride] = x

            # 如果存在deconv，则对经过conv的结果进行反卷积操作

            # 上面刚对x进行了下采样 x = self.blocks[i](x)，x的值改变了，为当前块的输出
            # 然后便进行上采样，这里没修改x的值
            # self.deblocks与blocks中都包含三个nn.Sequential()
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        # 将上采样结果在通道维度拼接
        # 每个上采样结果: B * 2C * H/2 * W/2
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        # 只有一块上采样，不需要拼接了
        elif len(ups) == 1:
            x = ups[0]

        # Fasle 如果上采样块比下采样块多，还需要进行多余的块执行尺寸不变的上采样
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # 将结果存储在spatial_features_2d中并返回
        data_dict['spatial_features_2d'] = x

        return data_dict
