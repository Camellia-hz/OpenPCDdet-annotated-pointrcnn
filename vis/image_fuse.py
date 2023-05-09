import cv2
import numpy as np

ori_img = cv2.imread("Screenshot from 2022-02-18 22-00-57.png")

H, W, _ = ori_img.shape

step = 20

plt_img = ori_img.copy()
for i in range(W):
    if i % 20 == 0:
        cv2.line(plt_img, (i, 0), (i, H), color=(150, 150, 150), thickness=1)
for i in range(H):
    if i % 20 == 0:
        cv2.line(plt_img, (0, i), (W, i), color=(150, 150, 150), thickness=1)

# h_coor = np.arange(0, H, step)
# w_coor = np.arange(0, W, step)
# coor = np.meshgrid(h_coor, w_coor)

# cv2.imshow("11", ori_img)
res = np.hstack((ori_img, plt_img))

cv2.imwrite("./res.png", res)
cv2.imshow("11", res)
cv2.waitKey(0)
