from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        """
        数据增强类
        Args:
            :param root_path: 根目录
            :param augmentor_configs: 增强器配置
            :param class_names: 类别名
            :param logger: 日志
        """
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        # 数据增强器队列
        self.data_augmentor_queue = []

        # 如果augmentor_configs是队列， 则将其整个（看pointpillars.yaml.DATA_AUGMENTOR包含
        # 两部分，分别为disable_aug_list和AUG_CONFIG_LIST ）赋予aug_config_list,否则仅将后面的
        # AUG_CONFIG_LIST赋给他，这部分仅有gt_sample

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list)  \
            else augmentor_configs.AUG_CONFIG_LIST
        # 将配置中的增强方法放入到data_augmentor_queue中，然后forward中逐个执行
        # 增强方法的具体函数定义在下面
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                # 将被禁止的增强操作跳过,augmentor_configs.DISABLE_AUG_LIST==['placeholder'],这部分不需要增强
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # 根据名称和配置，获取增强器, cug_cfg.NAME为 gt_sample的整个配置
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        """
        ground truth 采样增强， 调用dataset_sampler的DataBaseSampler处理 （from paper ： second）
        """
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        """
        随机翻转
        """
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        # 获取gt_boxes和points
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']

        # 确定翻转轴
        for cur_axis in config['ALONG_AXIS_LIST']:
            # assert如果它的条件返回错误，则终止程序执行
            # 如果翻转轴不在x, y内，则返回错误
            assert cur_axis in ['x', 'y']
            # 调用augmentor_utils中的函数翻转bos和点云
            # getattr用于返回某个对象的属性或者方法，
            # getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)
            # 上句为 获取augmentor_utils.py 中的 'random_flip_along_%s' % cur_axis方法
            # 方法的获得由 % cur_axis对 %s赋值得到

            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
        # 将翻转后的box和数据更新一下
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # def random_world_filp(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         return partial(self.random_world_flip, config=None)
    #     gt_boxes, points= data_dict['gt_boxes'], data_dict['points']
    #
    #     for cur_axis in config.ALONG_AXIS_LIST:
    #         assert  cur_axis in ['x', 'y']
    #         gt_boxes, points = getattr(augmentor_utils, 'random_filp_along_%s' %cur_axis)(
    #             gt_boxes, points
    #         )
    #     data_dict['gt_boxes'], data_dict['points']= gt_boxes, points
    #     return  data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        """随机旋转 使points和gt_boxes进行绕Z轴的旋转波动"""
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        # 获取点云范围
        rot_range = config['WORLD_ROT_ANGLE']
        # 如果旋转范围不是列表，则去正负
        # 将rot_range 变为列表，正负范围
        # [0, -39.68, -3, 69.12, 39.68, 1]
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        # 调用augmentor_utils中的函数旋转box和点云
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        """
        随机缩放
        """
        # partial 函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数
        # partial(func, func's args)
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        # 调用augmentor_utils中的函数缩放box和点云
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']  # [0.95, 1.05]
        )
        # 将放缩后的gt_boxes points重新赋给他
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        """
        随机图片翻转
        """
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        # 获取图像，深度图，3Dbox和2Dbox以及abrading信息
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        # 遍历翻转轴列表
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']  # np.fliplr只能水平翻转
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        offset_range = config['WORLD_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)

        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # 遍历增强队列，逐个增强器做数据增强
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        # 将方位角限制[-pi, pi]
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )

        # 将标定信息弹出
        if 'calib' in data_dict:
            data_dict.pop('calib')
        # 将地面信息弹出
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        # gt_boxes_mask参数在dataset.py中传入到data_dict中，用来删除我们不需要的目标分类的box
        # 筛选mask选中的信息，最后将mask信息删除，这里的mask是一个bool向量，对应到每个box上
        # true-->保留box，FALSE-->删除这个box
        # 如果给了gt_boxes_mask参数
        # data_dict = self.data_augmentor.forward(
        #                 data_dict={
        #                     **data_dict,
        #                     'gt_boxes_mask': gt_boxes_mask
        #                 }
        #             )
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
