from HBB_Soft_NMS_CPU.cpu_soft_nms import cpu_nms, cpu_soft_nms

import cv2
import numpy as np
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

"""cpu_nms
Format:
cpu_nms(dets, threshold)
dets = [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score] ...], type must np.array
threshold type float"""

#keep = cpu_nms(bboxes, 0.2)
#keep_bboxes = bboxes[keep]


"""cpu_soft_nms
Format:
cpu_soft_nms(dets, sigma, Nt, threshold, method)
dets = type np.array, [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score] ...]
method = type int,  [1: 'linear', 2: 'gaussian', 3: 'original NMS']
sigma = type float, only used when method==2
Nt = type float <IoU thr>
threshold = type float <score thr>  
"""
keep = cpu_soft_nms(bboxes, sigma=0.5, Nt=0.25, threshold=0.5, method=1)
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

cv2.imwrite('test_HBB_Soft_NMS_CPU_before.png', img)
cv2.imwrite('test_HBB_Soft_NMS_CPU_after.png', img1)
