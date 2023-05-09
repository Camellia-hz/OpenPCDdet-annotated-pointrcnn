import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        # POINT_FEATURE_ENCODING: {
        #     encoding_type: absolute_coordinates_encoding,
        #     used_feature_list: ['x', 'y', 'z', 'intensity'],
        #     src_feature_list: ['x', 'y', 'z', 'intensity'],
        # }
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range


    @property
    def num_point_features(self):
        #  getattr(self, self.point_encoding_config.encoding_type)(points=None)
        # 获得本类当中的self.point_encoding_config.encoding_type函数，该函数参数为points，默认为none
        # point_encoding_config=config, config为配置，其中的encoding_type为absolute_coordinates_encoding函数
        # 所以，实际上就是调用了absolute_coordinates_encoding函数
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        # use_lead_xyz为absolute_coordinates_encoding(self, points=None)中返回的true
        data_dict['use_lead_xyz'] = use_lead_xyz
       
        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            max_sweeps = self.point_encoding_config.max_sweeps
            idx = self.src_feature_list.index('timestamp')
            dt = np.round(data_dict['points'][:, idx], 2)
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt))-1, max_sweeps-1)]
            data_dict['points'] = data_dict['points'][dt <= max_dt]
        
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        # 如果是None, 那么point的特征直接取used_feature_list中包含的特征['x', 'y', 'z', 'intensity']
        # 四个特征
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features
        # 如果points不为空，那就要从基本的特征也就是xyz中加入used_feature_list 我们新自定义的点云特征
        # points中包含非常多的特征，需要根据需求，从里面取若干维度代表的特征，需求即是used_feature_list
        # 而src_feature_list保存的是点云所有特征定义, 且idx与特征所在维度对应
        # 对point_feature_list先取基本特征xyz坐标
        point_feature_list = [points[:, 0:3]]
        # 遍历我们的需求特征
        for x in self.used_feature_list:
            # 如果需求特征中有xyz, 直接跳过，因为point_feature_list已经有了
            if x in ['x', 'y', 'z']:
                continue
            # 不在的话，查找x这个特征在points中所在的维度
            idx = self.src_feature_list.index(x)
            # 将points中的维度特征加入到point_feature_list中
            point_feature_list.append(points[:, idx:idx+1])
        # 全部找完后，将point_feature_list的特征全部拼接起来
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True
