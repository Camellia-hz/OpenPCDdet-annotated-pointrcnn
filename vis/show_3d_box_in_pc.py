"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import cv2
import numpy as np
import pandas as pd
import copy

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=True):
    """
    绘制点云
    Args:
        points:点云
        gt_boxes:真值box （N, 7）
        ref_boxes:预测box （M, 7）
        ref_scores:预测分数 (M,)
        ref_labels:预测类别 (M,)
    """
    # 1.判断数据类型，并将数据从tensor转化为numpy的array
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def read_detection(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                  'bbox_right', 'bbox_bottom', 'width', 'height', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y',
                  ]
    #     df.loc[df.type.isin(['Truck', 'Van', 'Tram']), 'type'] = 'Car'
    #     df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    #    df = df[df['type']=='Car']
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    df.reset_index(drop=True, inplace=True)
    return df


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


from pcdet.utils import calibration_kitti


def get_calib(calib_file):
    # assert calib_file.exists()

    return calibration_kitti.Calibration(calib_file)


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]
    # rect_to_lidar means rectified to lidar
    # 由于xyz参数是基于2号相机坐标系下获得的，因此需要转换到点云坐标系下
    xyz_lidar = calib.rect_to_lidar(xyz_camera)

    new_r = -(r + np.pi / 2)
    # 将xyz中的z轴坐标由物体标注框的底部移动到标注框的中心点
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)
    # return np.concatenate([xyz_lidar, l, w, h, r], axis=-1)


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image


if __name__ == '__main__':
    img_id = 5

    velo = '/home/nathan/OpenPCDet/data/kitti/training/velodyne/%06d.bin' % img_id
    calib_file = '/home/nathan/OpenPCDet/data/kitti/training/calib/%06d.txt' % img_id
    label = '/home/nathan/OpenPCDet/data/kitti/training/label_2/%06d.txt' % img_id
    img = '/home/nathan/OpenPCDet/data/kitti/training/image_2/%06d.png' % img_id

    points = np.fromfile(velo, dtype=np.float32).reshape(-1, 4)
    calib = get_calib(calib_file)
    df = read_detection(label)

    data_gt = []
    index_mask = [3, 4, 5, 2, 1, 0, 6]
    img = cv2.imread(img)

    corners= []
    for o in range(len(df)):
        pc_box = [*df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']]]
        pc_box = np.array(pc_box)
        gt = pc_box[index_mask]
        # e+01 表示 10^1
        data_gt.append(gt)

        corners_3d_cam2 = compute_3d_box_cam2(
            *df.loc[o, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
        corners.append(corners_3d_cam2.T)
        # pts_2d, _ = calib.rect_to_img(corners_3d_cam2.T)
        #
        # img = draw_projected_box3d(img, pts_2d, color=(255, 0, 255), thickness=1)
    corners = np.array(corners)
    _, box = calib.corners3d_to_img_boxes(corners)
    img = draw_projected_box3d(img, box[0], color=(255, 0, 255), thickness=1)


    data_gt = np.array(data_gt)

    data_pc = boxes3d_kitti_camera_to_lidar(data_gt, calib)

    # img = draw_projected_box3d(cv2.imread(img), *data_gt)
    cv2.namedWindow("img")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(data_gt)
    print(data_pc)

    draw_scenes(points, gt_boxes=data_pc)
