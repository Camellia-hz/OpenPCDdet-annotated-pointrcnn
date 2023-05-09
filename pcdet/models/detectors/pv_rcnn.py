from .detector3d_template import Detector3DTemplate


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    """
    1、MeanVFE
    2、VoxelBackBone8x
    3、HeightCompression  高度方向堆叠
    4、VoxelSetAbstraction
    5、BaseBEVBackbone
    6、AnchorHeadSingle
    7、PointHeadSimple   Predicted Keypoint Weighting
    8、PVRCNNHead
    """

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        # anchor head损失
        loss_rpn, tb_dict = self.dense_head.get_loss()
        # point head损失(PKW 前背景分割)
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        # roi头损失
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
