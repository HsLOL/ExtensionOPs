from HBB_NMS_GPU.nms.gpu_nms import gpu_nms

import cv2
import numpy as np
import torch
bboxes = np.array([[110, 110, 210, 210,  0.88],
                   [100, 100, 200, 200,  0.99],
                   [100, 150, 200, 250,  0.66],
                   [250, 250, 350, 350,  0.77]], dtype=np.float32)
img = np.zeros((1000, 1000, 3), np.uint8)
img.fill(255)
for idx in range(len(bboxes)):
    single_bbox = bboxes[idx]
    point1 = (int(single_bbox[0]), int(single_bbox[1]))
    point2 = (int(single_bbox[2]), int(single_bbox[3]))
    cv2.rectangle(img, point1, point2, color=[255, 0, 0], thickness=2)

# gpu_nms(dets, threshold, device_id)
# dets = type np.array, [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score] ...]
# threshold = type float
# device_id = type int
keep = gpu_nms(bboxes, 0.25, device_id=1)
keep_bboxes = bboxes[keep]

print(keep)
print(keep_bboxes)

img1 = np.zeros((1000, 1000, 3), np.uint8)
img1.fill(255)

for i in range(len(keep_bboxes)):
    single_bbox = keep_bboxes[i]
    point1 = (int(single_bbox[0]), int(single_bbox[1]))
    point2 = (int(single_bbox[2]), int(single_bbox[3]))
    cv2.rectangle(img1, point1, point2, color=[0, 255, 0], thickness=2)

cv2.imwrite('test_HBB_NMS_GPU_before.png', img)
cv2.imwrite('test_HBB_NMS_GPU_after.png', img1)
