from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


# 生成pillar的类，在数据预处理的第三个函数 transform_points_to_voxels
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    # 把points输入，生成voxels, coordinates, num_points
    def generate(self, points):
        # spconv包的版本，spconv_ver是_voxel_generator.generate(points)
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


# 数据预处理类
class DataProcessor(object):
    """
    数据预处理类
    Args:
        processor_configs: DATA_CONFIG.DATA_PROCESSOR
        point_cloud_range： 点云范围
        training：训练模式
    """

    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        # grid或voxel或pillar的size
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None
        # 工厂模式，根据不同的配置，只需要增加相应的方法即可实现不同的调用
        # cur_cfg=["mask_points_and_boxes_outside_range", "shuffle_points", "transform_points_to_voxels"]
        # getattr(self, cur_cfg.NAME)  cur_cfg.NAME分别是数据处理的三个函数，通过getattr可以调用self自身类下的这几个函数
        # 需要使用几个数据处理函数，就直接在processor_configs加入即可
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            # 在forward函数中调用
            self.data_processor_queue.append(cur_processor)
        # 最终，self.data_processor_queue 中依次包含了三个数据处理函数

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """
        移除超出point_cloud_range的点
        """
        # 偏函数是将所要承载的函数作为partial()函数的第一个参数，
        # 原函数的各个参数依次作为partial()函数后续的参数
        # 以便函数能用更少的参数进行调用
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            # mask为bool值，而且是个向量，是针对points中每个点是否在范围内的向量，将x和y超过规定范围的点设置为0
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            # 根据mask提取点
            data_dict['points'] = data_dict['points'][mask]
        # 当data_dict存在gt_boxes并且REMOVE_OUTSIDE_BOXES=True并且处于训练模式
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            # mask为bool值，将box角点在范围内点个数大于最小阈值的设置为1
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict
    # def mask_points_and_boxes_outside_range(self, data_dict, config=None):
    #     if data_dict is None:
    #         return partial(self.mask_points_and_boxes_outside_range, config=None)
    #
    #     if data_dict['points', None] is not None:
    #         mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
    #         data_dict['points']= data_dict['points'][mask]
    #     if data_dict['gt_boxes', None] is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
    #         mask = box_utils.mask_boxes_outside_range_numpy(
    #             data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=
    #         )

    def shuffle_points(self, data_dict=None, config=None):
        """将点云打乱"""
        if data_dict is None:
            return partial(self.shuffle_points, config=config)
        # self.mode = train or test
        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            # 生成随机序列
            # points: (M, 3 + C) 第一维是点，将点的idx进行随机排布
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points
        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        """
        将点云转换为voxel,调用spconv的VoxelGeneratorV2
        """
        if data_dict is None:
            # kitti截取的点云范围是[0, -39.68, -3, 69.12, 39.68, 1]
            # 得到[69.12, 79.36, 4]/[0.16, 0.16, 4] = [432, 496, 1]
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)
        # 这其实相当于定义一个函数：将函数封装在self.voxel_generator当中
        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                # 给定每个voxel的长宽高  [0.16, 0.16, 4]
                vsize_xyz=config.VOXEL_SIZE,  # [0.16, 0.16, 4]
                # 给定点云的范围 [  0.  -40.   -3.   70.4  40.    1. ]
                coors_range_xyz=self.point_cloud_range,
                # 给定每个点云的特征维度，这里是x，y，z，r 其中r是激光雷达反射强度
                num_point_features=self.num_point_features,
                # 给定每个pillar/voxel中有采样多少个点，不够则补0
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,  # 32
                # 最多选取多少个voxel，训练16000，推理40000
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],  # 16000
            )

        # 使用spconv生成voxel输出
        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)

        # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
        # voxels代表了每个生成的voxel数据，维度是[M, 32, 4]
        # coordinates代表了每个生成的voxel所在的zyx轴坐标，维度是[M,3],其中z恒为0
        # num_points代表了每个生成的voxel中有多少个有效的点维度是[M,1]，因为不满32会被0填充
        voxels, coordinates, num_points = voxel_output

        # False
        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        """
       采样点云，多了丢弃，少了补上
       """
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        # 如果采样点数 < 点云点数
        if num_points < len(points):
            # 计算点云深度
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            # 根据深度构造mask
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            # 如果采样点数 > 远点数量
            if num_points > len(far_idxs_choice):
                # 在近点中随机采样，因为近处稠密
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                # 如果远点不为0,则将采样的近点和远点拼接，如果为0,则直接返回采样的近点
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            # 如果采样点数 > 远点数量， 则直接随机采样
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            # 将点打乱
            np.random.shuffle(choice)
        # 如果采样点数 > 点云点数, 则随机采样点补全点云
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                # 　随机采样缺少的点云索引
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                # 拼接索引
                choice = np.concatenate((choice, extra_choice), axis=0)
            # 将索引打乱
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        """
        计算网格范围
        """
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        """降采样深度图"""
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)
        # skimage中类似平均池化的操作，进行图像将采样
        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 在for循环中逐个流程处理，最终都放入data_dict中
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
