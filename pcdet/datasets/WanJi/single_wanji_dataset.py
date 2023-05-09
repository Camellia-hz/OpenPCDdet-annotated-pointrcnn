import copy
import pickle
import os
import open3d as o3d
import numpy as np
from pathlib import Path

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate

class SingleWanjiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path: dataset路径
            dataset_cfg: 数据集配置文件 yaml tools/cfgs/dataset_configs/xxxxx.yaml
            class_names: 分类标签
            training:  训练/评估 mode
            logger: 日志
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 传递参数 训练集是train 验证集是val
        # DATA_SPLIT: {
        #     'train': train,
        #     'test': val
        # }
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # 第一句在类别列表中 获取对应类别的索引  eg： 0-->bus
        self.class_id2name = {k:v for k, v in enumerate(self.class_names)}
        # root_path的路径是../data/WanJi/
        split_dir = os.path.join(self.root_path, 'multi_split', (self.split + '.txt'))

        # 得到选取的.txt文件下的序列号，组成sample_id_list
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None
        self.logger.info(split_dir)

        # 创建用于存放wanji信息的空列表
        self.wanji_infos = []
        # 调用函数，加载kitti数据，mode的值为train或者val
        self.include_data(self.mode)
        
    def include_data(self, mode):
        # 如果日志信息存在，则加入“Loading SingleWanji dataset的信息” 便于后续查看
        self.logger.info('Loading SingleWanji dataset.')
        #  创建新字典，用于存放信息
        wanji_infos = {}

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            # 以kitti为例
            """
                    INFO_PATH：{
                    "train":[kitti_infos_train.pkl],
                    "test":[kitti_infos_val.pkl],
                    }
            """
            # 生成的pkl文件
            # root path 的路径是/data/kitti/
            # 则info_path的路径是：/data/kitt/kitti_infos_train.pkl | /data/kitt/kitti_infos_val.pkl
            info_path = os.path.join(self.root_path, info_path)
            # 如果一个文件不存在，则跳过，执行下一个
            if not os.path.exists(info_path):
                continue
            # 打开该文件，获得文件句柄f
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                # kitti中使用extend(infos)是因为kitti_infos是列表，字典添加键值对使用update
                wanji_infos.update(infos)

        self.length = 0
        for seq in self.sample_seq_list:
            for pc, an in zip(wanji_infos[seq]['point_cloud'], wanji_infos[seq]['annos']):
                self.wanji_infos.append({'point_cloud': pc, 'annos': an})
            self.length += wanji_infos[seq]['frame_num']
        
        self.logger.info('Total samples for SingleWanji dataset: %d' % (self.length))
        
    def get_label(self, sequence_name, idx):
        # 读取时需要使用format规范csv文件的名称格式，左边填充两位0，保证与kitti标签名称格式对齐
        label_file = os.path.join(self.root_path, 'label_7', sequence_name, '{:0>4d}.csv'.format(idx))
        # print(label_file)
        # E:\WANJI_dataset\Wanji_dataset_0621\Lidar\20211215151804\label_7\00idx.csv
        # 万集的点云标签格式是csv
        assert os.path.exists(label_file)
            
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        # 读取某个csv标签文件的全部标注信息
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(',')
            # eval(x)将csv保存字符串格式转换为数字
            class_id = int(eval(line_list[1]))
            # 坐标，大小都转换单位为m
            loc = [eval(x)/100 for x in line_list[2:5]]
            size = [eval(x)/100 for x in line_list[7:10]]
            # 航向角
            rots = eval(line_list[6])
            # 将目标框信息添加到gt_boxes
            gt_boxes.append(np.concatenate([loc, size, [rots/180*np.pi]]))
            # 取目标框对应的类别标签
            gt_names.append(self.class_id2name[class_id])
        # 最后，将获得的gt_label信息以矩阵返回
        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)
    
    def get_lidar(self, sequence_name, idx):
        lidar_file = os.path.join(self.root_path, 'pcd', sequence_name, '{:0>4d}.pcd'.format(idx))
        assert os.path.exists(lidar_file)
        # 读取pcd格式的点云数据
        # 每一帧的点云数据是pcd文件，每行4位数表示一个点（x，y，z，强度），单位是m；
        pcd = o3d.t.io.read_point_cloud(lidar_file)
        position = pcd.point["positions"].numpy()
        intensity = pcd.point["intensity"].numpy()
        return np.concatenate([position, intensity], axis=1)
        
    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        # 仅在20211215161104中含有split文件，分出来train和test
        # 选择其中的一个文件
        split_dir = os.path.join(self.root_path, 'multi_split', self.split+'.txt')
        # 构造文件（训练、测试样本）列表
        self.sample_seq_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

    def __len__(self):
        # 等于返回训练帧的总个数
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_seq_list) * self.total_epochs

        return self.length
    
    def __getitem__(self, index):
        """
               从pkl文件中获取相应index的info，然后根据info['point_cloud']['lidar_idx']确定帧号，进行数据读取和其他info字段的读取
               初步读取的data_dict,要传入prepare_data（dataset.py父类中定义）进行统一处理，然后即可返回
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.wanji_infos)
        # 拷贝第index帧中的新到到info中
        # 因为pkl文件是将所有帧的点云数据放在一起，所以使用idx取pkl中的一个帧
        info = copy.deepcopy(self.wanji_infos[index])
        # 获取采样的序列号，在train.txt文件里的数据序列号
        sequence_name = info['point_cloud']['sequence_idx']
        sample_idx = info['point_cloud']['lidar_idx']
        # points = self.get_lidar(sample_idx)

        # 定义输入数据的字典包含帧id和sequence_name
        input_dict = {
            'frame_id': sample_idx,
            'sequence_name': sequence_name
        }
        # 获取该帧当中的点的信息
        points = self.get_lidar(sequence_name, sample_idx)

        # annos里面的信息都是在生成pkl文件的代码当中
        # 在kitti当中，pkl文件包含了所有帧的数据，其中每一帧的信息如下：
        """
                其中每一帧都包含了四个信息：
                {
                    "point_cloud" : {"num_feature" : 4, "lidar_idx" : "xxxxxx"},

                    "image" : {"image_idx" : "xxxxxx", "image_shape" : array([xxx,xxx],dtype=int32)},

                    "calib" : {"P2" : 代表2号相机内参矩阵,
                                R0_rect : R0_rect 为0号相机的修正矩阵,
                                "Tr_velo_to_cam":为velodyne到camera的矩阵 大小为3x4，包含了旋转矩阵 R 和 平移向量 t},

                    "annos" : {'name', 'truncated', 'occluded', 'alpha', 'bbox', 
                                'dimensions', 'location', 'rotation_y', 'score', 'difficulty', 
                                'index', 'gt_boxes_lidar', 'num_points_in_gt'}
                }
        """
        if 'annos' in info:
            # 获取该帧信息中的 annos
            annos = info['annos']
            # 获取该帧信息中的有效物体object(N个)的名称
            gt_names = annos['name']
            # 得到有效物体object(N个)的gt_boxes_lidar信息，应该包含位置、大小和角度信息（N,3）,(N,3),(N)
            gt_boxes_lidar = annos['gt_boxes_lidar']
            # 将新的键值对 添加到输入的字典中
            input_dict.update({
                'points': points,
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        # 将输入数据送入prepare_data进一步处理，形成训练数据
        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    def prepare_data(self, data_dict):
        """
        接受统一坐标系下的数据字典（points，box和class），进行数据筛选，数据预处理，包括数据增强，点云编码等
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        
        # save = {}
        # save.update({'sequence_name': data_dict['sequence_name'], 'frame_id': data_dict['frame_id']})
        # save.update({'before': data_dict['gt_names']})
                    
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            # 返回一个bool数组，记录自定义数据集中ground_truth_name列表在不在我们需要检测的类别列表self.class_name里面
            # 比如kitti数据集中data_dict['gt_names']='car','person','cyclist'
            # 对于在data_dict['gt_names']]中的所有类别的box，我们只要分类目标的box即可
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            # 数据增强 传入字典参数，**data_dict是将data_dict里面的key-value对都拿出来
            # 下面在原数据的基础上增加gt_boxes_mask，构造新的字典传入data_augmentor的forward函数
            # 进行数据增强，数据增强的其余参数在__init__中，已经对self.data_augmentor传入了数据增强的类，并给了参数
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
                    
        if data_dict.get('gt_boxes', None) is not None:
            # 筛选需要检测的gt_boxes
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            # 根据selected，选取需要的gt_boxes和gt_names 仅需要class_name中的gtbox
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]

            # 将当帧数据的gt_names中的类别名称对应到class_names的下标
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            # 将类别index信息放到每个gt_boxes的最后，对每个gt_box赋予class_name中的标签
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            # 完成更新
            data_dict['gt_boxes'] = gt_boxes

        # 对点的特征维度进行编码
        if data_dict.get('points', None) is not None: # (119342, 4)
            data_dict = self.point_feature_encoder.forward(data_dict)

        # 对点云进行预处理，包括移除超出point_cloud_range的点、 打乱点的顺序以及将点云转换为voxel等方法，使用哪些具体看配置文件
        data_dict = self.data_processor.forward( # add multi_frame
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0: 
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict
    
    def evaluation(self, det_annos, class_names, logger, **kwargs):
        if 'annos' not in self.wanji_infos[0].keys():
            return None, {}
        
        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval_new as kitti_eval
            from ..kitti import kitti_utils

            eval_det_annos = kitti_utils.transform_det_annotations(eval_det_annos)
            eval_gt_annos = kitti_utils.transform_gt_annotations(eval_gt_annos)
            
            ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names
            )
            return ap_dict
        
        def kitti_range_eval(det_annos, gt_annos, logger):
            from ..kitti.kitti_object_eval_python import eval_new as kitti_eval
            from ..kitti import kitti_utils

            eval_det_annos = kitti_utils.transform_det_range_annotations(det_annos)
            eval_gt_annos = kitti_utils.transform_gt_range_annotations(gt_annos)
            ap = []
            for i in range(len(eval_det_annos)):
                ap_dict = kitti_eval.get_official_eval_result(
                    gt_annos=eval_gt_annos[i], dt_annos=eval_det_annos[i], current_classes=class_names
                )
                # kitti_eval.get_mine_eval_result(eval_gt_annos[i], dt_annos=eval_det_annos[i], logger=logger)
                ap.append(ap_dict)
            return ap

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.wanji_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
            range_ap_dict = kitti_range_eval(eval_det_annos, eval_gt_annos, logger)
        else:
            raise NotImplementedError

        return ap_dict, range_ap_dict, None
    
    def get_total_infos(self, num_workers=4, has_label=True, sample_seq_list=None, num_features=5):
        import concurrent.futures as futures
        def process_single_scene(sample_seq_idx):
            print('sample_idx: %s' % (sample_seq_idx))
            sample_id_list = os.listdir(os.path.join(self.root_path, 'label_7', sample_seq_idx))
            length = len(sample_id_list)
            info = {}
            info[sample_seq_idx] = {}
            info[sample_seq_idx]['point_cloud'] = []
            info[sample_seq_idx]['annos'] = []
            info[sample_seq_idx]['frame_num'] = length
            for sample_idx, _ in enumerate(sample_id_list):
                pc_info = {'num_features': num_features, 'lidar_idx': sample_idx, 'sequence_idx': sample_seq_idx}
                info[sample_seq_idx]['point_cloud'].append(pc_info)
                if has_label:
                    annotations = {}
                    gt_boxes_lidar, name = self.get_label(sample_seq_idx, sample_idx)
                    annotations['name'] = name
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                    info[sample_seq_idx]['annos'].append(annotations)
            return info
        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        # print(sample_id_list)
        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(4) as executor:
            infos = executor.map(process_single_scene, sample_seq_list)
        res = {}
        for item in list(infos):
            res.update(item)
        return res

    # 用trainfile的groundtruth产生groundtruth_database，
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强，数据库采样增强
    def create_gt_db(self, used_classes=None, split='train'):
        import torch
        # 如果是“train”，创建的路径是  ../gt_database
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('multi_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        # 读取wanji_infos里的每个info信息，一个info是一帧的数据,self.wanji_infos保存了pkl中的所有数据信息
        for k in range(self.length):
            # 输出的是第几个样本 如7/780
            print('gt_database sample: %d/%d' % (k + 1, self.length))
            # 取当前帧的信息 info
            info = self.wanji_infos[k]
            # 取里面的样本序列
            sample_idx = info['point_cloud']['lidar_idx']
            sample_seq = info['point_cloud']['sequence_idx']
            # 获取当前pcd点云样本中的的点
            # points是一个数组（M,4）
            points = self.get_lidar(sample_seq, sample_idx)
            # 读取注释信息
            annos = info['annos']
            # name的数据是['car','car','pedestrian'...'dontcare'...]表示当前帧里面的所有物体objects
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']
            # num_obj是有效物体的个数，为N取第0维的维度即可
            num_obj = gt_boxes.shape[0]
            # 返回每个box中点云索引
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
            
            for i in range(num_obj):
                filename = '{}_{}_{}_{}.bin'.format(sample_seq, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    # 把db_info信息添加到 all_db_infos字典里面
                    if names[i] in all_db_infos:
                        # 如果存在该类别则追加
                        all_db_infos[names[i]].append(db_info)
                    else:
                        # 如果不存在该类别则新增
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
            
def create_total_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    """
            生成.pkl文件（包括train、test、val），提前读取点云格式和label
    """
    dataset = SingleWanjiDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split = os.path.join(dataset_cfg.DATA_PATH, 'multi_split', 'train.txt')
    test_split = os.path.join(dataset_cfg.DATA_PATH, 'multi_split', 'val.txt')
    # 分别读取训练集、验证集
    sample_seq_list = [x.strip() for x in open(train_split).readlines()] if os.path.exists(train_split) else None
    test_list = [x.strip() for x in open(test_split).readlines()] if os.path.exists(test_split) else None
    
    sample_seq_list.extend(test_list)
    
    # num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
    filename = os.path.join(save_path, 'multi_infos.pkl')
    # print('------------------------Start to generate data infos------------------------')
    # dataset.set_split('train')
    # 下面是得到sample_seq_list中序列相关的所有点云数据的信息，并且进行保存
    wanji_infos = dataset.get_total_infos(
        class_names, has_label=True, num_features=5, sample_seq_list=sample_seq_list
    )
    with open(filename, 'wb') as f:
        pickle.dump(wanji_infos, f)
    print('Wanji info file is save to %s' % filename)
    
    print('------------------------Start create groundtruth database for data augmentation------------------------')
    # 用trainfile产生groundtruth_database
    # 只保存训练数据中的gt_box及其包围点的信息，用于数据增强
    # dataset.set_split('train')
    # dataset.create_gt_db(split='train')
    print('------------------------Data preparation done------------------------')
         

if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
    # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    ROOT_DIR = '/media/disk/ssd_01/Wanji_dataset_0621/Lidar/seq_data'
    
    create_total_infos(
        dataset_cfg=dataset_cfg,
        class_names=['bus', 'car', 'bicycle', 'pedestrian', 'tricycle', 'semitrailer', 'truck'],
        data_path=os.path.join(ROOT_DIR),
        save_path=os.path.join(ROOT_DIR , 'info') ,
    )
    
    # dataset = MultiWanjiDataset(
    #     dataset_cfg=dataset_cfg, 
    #     class_names=['bus', 'car', 'bicycle', 'pedestrian', 'tricycle', 'semitrailer', 'truck'], 
    #     root_path=os.path.join(ROOT_DIR),
    #     training=False, logger=common_utils.create_logger()
    # )
    # python -m pcdet.datasets.multi_wanji.multi_wanji_dataset create_total_infos tools/cfgs/dataset_configs/multi_wanji_dataset.yaml
