import numpy as np
import torch
import torch.nn as nn

from ....ops.iou3d_nms import iou3d_nms_utils


class ProposalTargetLayer(nn.Module):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        # roi的box参数、 roi对应GT的box、 roi和GT的最大iou、 roi的类别预测分数、  roi的预测类别
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict
        )

        # regression valid mask
        # 得到需要计算回归损失的的roi的mask，其中iou大于0.55就是
        # 在self.sample_rois_for_rcnn（）中定义为真正属于前景的roi
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()

        # classification label
        # 对iou大于0.6的roi进行分类，忽略iou属于0.45到0.6之间roi的loss计算   PointRCNN
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & \
                          (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            # 将iou属于0.45到0.6之前的roi的类别置-1，不计算loss
            batch_cls_labels[ignore_mask > 0] = -1

        # 3D Intersection-over-Union (IoU) quality-aware confidence prediction
        # 对iou大于0.75的置信度预测应为1,小于0.25的预测置信度为0，
        # 0.25到0.75之间的使用(iou-0.25)/0.5作为置信度预测结果 PV-RCNN
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH  # 0.25
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH  # 0.75
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError
        """
        'rois': batch_rois,                 roi的box的7个参数                    shape（batch， 128, 7）
        'gt_of_rois': batch_gt_of_rois,     roi对应的GTbox的8个参数，包含类别      shape（batch， 128, 8）
        'gt_iou_of_rois': batch_roi_ious,   roi个对应GTbox的最大iou数值           shape（batch， 128）
        'roi_scores': batch_roi_scores,     roi box的类别预测分数                shape（batch， 128）
        'roi_labels': batch_roi_labels,     roi box的类别预测结果              shape（batch， 128）
        'reg_valid_mask': reg_valid_mask,   需要计算回归损失的roi            shape（batch， 128）
        'rcnn_cls_labels': batch_cls_labels 需要计算分类损失的roi             shape（batch， 128）
        """
        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """

        batch_size = batch_dict['batch_size']
        # 第一阶段生成的每个roi的位置和大小 （batch， num_of_roi, 7） 点云坐标系中的 xyzlwhθ
        rois = batch_dict['rois']
        # 第一阶段预测的每个roi的类别置信度 roi_score （batch， num_of_roi）
        roi_scores = batch_dict['roi_scores']
        # 第一阶段预测的每个roi的类别 roi_score （batch， num_of_roi）
        roi_labels = batch_dict['roi_labels']
        # gt_boxes （batch， num_of_GTs, 8） (x, y, z, l, w, h, heading, class)
        gt_boxes = batch_dict['gt_boxes']

        code_size = rois.shape[-1]  # box编码个数：7
        # 初始化处理结果的batch矩阵，后续将每帧处理的结果放入此处。batch_rois （batch， 128, 7）
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        # batch_gt_of_rois （batch， 128, 8） GTBox为7个box的参数和1个类别
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        # batch_gt_of_rois （batch， 128）  ROI和GT的最大iou
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        # batch_roi_scores （batch， 128） ROI预测类别置信度
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        # batch_roi_labels （batch， 128） ROI预测的类别
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        # 逐帧处理
        for index in range(batch_size):
            # 得到当前帧的roi、gt、roi的预测类别、roi类别的置信度
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]

            k = cur_gt.__len__() - 1
            # 从GT中取出结果，因为之前GT中以一个batch中最多的GT数量为准，
            # 其他不足的帧中，在最后填充0数据。这里消除填充的0数据
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]  # （num_of_GTs, 8）
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            # 进行iou匹配的时候，只有roi的预测类别与GT相同时才会匹配该区域的roi到该区域的GT上
            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                # 　其中max_overlaps包含了每个roi和GT的最大iou数值，gt_assignment得到了每个roi对应的GT索引
                # max_overlaps(512, ) gt_assignment(512,)
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )

            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            # sampled_inds包含了从前景和背景采样的roi的索引
            """
            此处的背景采样不是意义上的背景，而是那些iou与GT小于0.55的roi，对这些roi进行采样
            sampled_inds(128,) roi的前背景采样
            """
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            # 将当前帧中被选取的roi放入batch_rois中   cur_roi[sampled_inds]  shape :（len(sampled_inds), 7）
            batch_rois[index] = cur_roi[sampled_inds]
            # 将当前帧中被选取的roi的类别放入batch_roi_labels中
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            # 将当前帧中被选取的roi与GT的最大iou放入batch_roi_ious中
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            # 将当前帧中被选取的roi的类别最大预测分数放入batch_roi_scores中
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            # 将当前帧中被选取的roi的GTBox参数放入batch_gt_of_rois shape （batch, 128, 8）
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
        # 返回一帧中选取的roi预测的box参数、roi对应GT的box、roi和GT的最大iou、roi的类别预测置信度、roi的预测类别
        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        """此处的背景采样不是意义上的背景，而是那些iou与GT小于0.55的roi，对这些roi进行采样"""
        # sample fg, easy_bg, hard_bg
        # 每帧点云中最多有多少个前景roi和属于前景roi的最小thresh
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)

        # 从512个roi中，找出其与GT的iou大于fg_thresh的那些roi索引
        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        # 将roi中与GT的iou小于0.1定义为简单背景，并得到在roi中属于简单背景的索引
        easy_bg_inds = ((max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(
            -1)
        # 将roi中与GT的iou大于等于0.1小于0.55的定义为难背景，并得到在roi中属于难背景的索引
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) &
                        (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)

        # numel就是"number of elements"的简写。numel()可以直接返回int类型的元素个数
        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
        # 如果该帧中，前景的roi大于0,并且背景的roi也大于0
        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg   采样前景，选取fg_rois_per_image、fg_num_rois的最小数值
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            # 将所有属于前景点的roi打乱，使用np.random.permutation()函数
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            # 直接取前N个roi为前景，得到被选取的前景roi在所有roi中的索引
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            # 背景采样，其中前景采样了64个，背景也采样64个，保持样本均衡，如果不够用负样本填充
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            # 其中self.roi_sampler_cfg.HARD_BG_RATIO控制了所有背景中难、简单背景的比例
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError
        # 将前景roi和背景roi的索引拼接在一起
        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        # 如果该帧中，难背景（iou与GT大于等于0.1小于0.55）和简单背景（iou与GT小于0.1）都大于0
        # 则使用hard_bg_ratio控制难、简单样本的采样比例
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:

            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        # 如果该帧中，难背景的数量大于0;简单背景的数量等于0，则所有背景都从难背景中采样
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        # 如果该帧中，难背景的数量等于0;简单背景的数量大于0，则所有背景都从简单背景中采样
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        # （512, ）用于存储所有roi与GT的最大iou数值
        max_overlaps = rois.new_zeros(rois.shape[0])
        # （512, ）用于存储所有roi与GT拥有最大iou的GT索引
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
        # 逐类别进行匹配操作
        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            # 取出预测结果属于当前类别的roi mask
            roi_mask = (roi_labels == k)
            # 得到当前GTs中属于当前类别mask
            gt_mask = (gt_labels == k)
            # 如果当前的预测结果有该类别并且GTs中也有该类别
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                # 根据mask索引roi中当前处理的类别 shape ：（num_of_class_specified_roi, 7）
                cur_roi = rois[roi_mask]
                # 根据mask索引当前GTs中属于当前正在处理的类别 shape ：（num_of_class_specified_GT, 7）
                cur_gt = gt_boxes[gt_mask]
                # 得到GT中属于当前类别的索引 shape :（num_of_class_specified_GT, ）
                original_gt_assignment = gt_mask.nonzero().view(-1)
                # 计算指定类别下 roi和GT之间的3d_iou shape :
                # （num_of_class_specified_roi, num_of_class_specified_GT）
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)  # (M, N)
                # 取出每个roi与当前GT最大的iou数值和最大iou数值对应的GT索引
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                # 将该类别最大iou的数值填充进max_overlaps中
                max_overlaps[roi_mask] = cur_max_overlaps
                # 将该类别roi与GT拥有最大iou的GT索引填充入gt_assignment中
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
