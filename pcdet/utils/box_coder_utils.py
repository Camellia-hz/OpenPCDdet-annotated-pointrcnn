import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        """
        loss中anchor和gt的编码与解码
        7个参数的编码的方式为
            针对 xyz的回归为平移+缩放
            针对长宽高的回归为尺度缩放 log
            针对航向角使用sin 防止0和2pi误检的情况发生
            ∆x = (x^gt − xa^da)/d^a , ∆y = (y^gt − ya^da)/d^a , ∆z = (z^gt − za^ha)/h^a
            ∆w = log (w^gt / w^a) ∆l = log (l^gt / l^a) , ∆h = log (h^gt / h^a)
            ∆θ = sin(θ^gt - θ^a)
        """
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        # 航向角如果使用sin, 则需要多预测一个, 因为预测是正向还是反向
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            编码是train时候使用, 将∆x等编码出来，作为真正的label与得到的预测值进行loss
        Returns:
            得到boxes(labels与各类别anchor的回归属性，然后与卷积预测的值求loss)
        """
        # 截断anchors的[dx,dy,dz]，每个anchor_box的l, w, h数值如果小于1e-5则为1e-5
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        # 截断boxes的[dx,dy,dz] 每个GT_box的l, w, h数值如果小于1e-5则为1e-5
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)
        # If split_size_or_sections is an integer type, then tensor will be split into equally sized chunks (if possible).
        # Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.

        # 这里指torch.split的第二个参数   torch.split(tensor, split_size, dim=)  split_size是切分后每块的大小，不是切分为多少块！，多余的参数使用*cags接收
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)
        # 计算anchor对角线长度
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)

        # 计算loss的公式，Δx,Δy,Δz,Δw,Δl,Δh,Δθ
        # ∆x = (x^gt − xa^da)/diagonal
        xt = (xg - xa) / diagonal
        # ∆y = (y^gt − ya^da)/diagonal
        yt = (yg - ya) / diagonal
        # ∆z = (z^gt − za^ha)/h^a
        zt = (zg - za) / dza
        # ∆l = log(l ^ gt / l ^ a)
        dxt = torch.log(dxg / dxa)
        # ∆w = log(w ^ gt / w ^ a)
        dyt = torch.log(dyg / dya)
        # ∆h = log(h ^ gt / h ^ a)
        dzt = torch.log(dzg / dza)
        # False
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]  # Δθ

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: 网络的预测结果
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: 剩余锚框的本身的位置与大小
        Returns:
            返回预测得到的gt
        """
        # 这里指torch.split的第二个参数   torch.split(tensor, split_size, dim=)

        # split_size是切分后每块的大小，不是切分为多少块！，多余的参数使用*cags接收
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        # 分割编码后的box PointPillar为False
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        # 计算anchor对角线长度
        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)  # (B, N, 1)-->(1, 321408, 1)
        # loss计算中anchor与GT编码的运算:g表示gt，a表示anchor
        # ∆x = (x^gt − xa^da)/diagonal --> x^gt = ∆x * diagonal + x^da
        # 下同
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za
        # ∆l = log(l^gt / l^a)的逆运算 --> l^gt = exp(∆l) * l^a
        # 下同
        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        # 如果角度是cos和sin编码，采用新的解码方式 PointPillar为False
        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            # rts = [rg - ra] 角度的逆运算
            rg = rt + ra
        # PointPillar无此项
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            另一种编码方式, pointpillars使用的是第一种
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        # 每个gt_box的长宽高不得小于 1*10^-5，这里限制了一下
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)
        # 这里指torch.split的第二个参数   torch.split(tensor, split_size, dim=)  split_size是切分后每块的大小，
        # 不是切分为多少块！，多余的参数使用*cags接收。dim=-1表示切分最后一个维度的参数
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        # 上面分割的是gt_box的参数，下面分割的是每个点的参数
        xa, ya, za = torch.split(points, 1, dim=-1)
        # True 这里使用基于数据集求出的每个类别的平均长宽高
        if self.use_mean_size:
            # 确保GT中的类别数和长宽高计算的类别是一致的
            assert gt_classes.max() <= self.mean_size.shape[0]

            """
            各类别的平均长宽高
            车：      [3.9, 1.6, 1.56],
            人：      [0.8, 0.6, 1.73],
            自行车：   [1.76, 0.6, 1.73]
            """

            # 根据每个点的类别索引，来为每个点生成对应类别的anchor大小 这个anchor来自于数据集中该类别的平均长宽高
            point_anchor_size = self.mean_size[gt_classes - 1]
            # 分割每个生成的anchor的长宽高
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            # 计算每个anchor的底面对角线距离
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            # 计算loss的公式，Δx,Δy,Δz,Δw,Δl,Δh,Δθ
            # 以下的编码操作与SECOND、Pointpillars中一样
            # 坐标点编码
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            # 长宽高的编码
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
            # 角度的编码操作为 torch.cos(rg)、torch.sin(rg) #
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        # 返回时，对每个GT_box的朝向信息进行了求余弦和正弦的操作
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:
        """
        # 这里指torch.split的第二个参数   torch.split(tensor, split_size, dim=)  split_size是切分后每块的大小，
        # 不是切分为多少块！，多余的参数使用*cags接收。dim=-1表示切分最后一个维度的参数
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        # 得到每个点在点云中实际的坐标位置
        xa, ya, za = torch.split(points, 1, dim=-1)
        # True 这里使用基于数据集求出的每个类别的平均长宽高
        if self.use_mean_size:
            # 确保GT中的类别数和长宽高计算的类别是一致的
            assert pred_classes.max() <= self.mean_size.shape[0]
            # 根据每个点的类别索引，来为每个点生成对应类别的anchor大小
            point_anchor_size = self.mean_size[pred_classes - 1]
            # 分割每个生成的anchor的长宽高
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            # 计算每个anchor的底面对角线距离
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            # loss计算中anchor与GT编码的运算:g表示gt，a表示anchor
            # ∆x = (x^gt − xa^da)/diagonal --> x^gt = ∆x * diagonal + x^da
            # 下同
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za
            # ∆l = log(l^gt / l^a)的逆运算 --> l^gt = exp(∆l) * l^a
            # 下同
            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
            # 角度的解码操作为 torch.atan2(sint, cost) #
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)
        # 根据sint和cost反解出预测box的角度数值
        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]

        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
