# coding:utf-8
import numpy as np


def soft_nms(boxes, sigma=0.5, IoU_thr=0.25, score_thr=0.5, method='linear'):
    # method = ['linear', 'gaussian', 'nms']
    """Pure Python Soft-NMS baseline."""
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
    # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

    # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

    # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
    # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 'linear': # linear
                        if ov > IoU_thr:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 'gaussian': # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > IoU_thr:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                    # print(boxes[:, 4])

            # if box score falls below threshold, discard the box by swapping with last box
            # update N
                    if boxes[pos, 4] < score_thr:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1
    keep = [i for i in range(N)]
    return keep


if __name__ == '__main__':
    import cv2
    # boxes = np.array([[100, 100, 150, 168, 0.63],
    #                   [166, 70, 312, 190, 0.55],
    #                   [221, 250, 389, 500, 0.79],
    #                   [12, 190, 300, 399, 0.9],
    #                   [28, 130, 134, 302, 0.3]])
    bboxes = np.array([[110, 110, 210, 210, 0.88],
                      [100, 100, 200, 200, 0.99],
                      [100, 150, 200, 250, 0.66],
                      [250, 250, 350, 350, 0.77]])

    img = np.zeros((1000, 1000, 3), np.uint8)
    img.fill(255)

    img1 = np.zeros((1000, 1000, 3), np.uint8)
    img1.fill(255)

    for idx, bbox in enumerate(bboxes):
        single_bbox = bboxes[idx]
        point1 = (int(single_bbox[0]), int(single_bbox[1]))
        point2 = (int(single_bbox[2]), int(single_bbox[3]))
        cv2.rectangle(img, point1, point2, color=[255, 0, 255], thickness=2)
    cv2.imwrite('before.png', img)

    keep = soft_nms(bboxes)
    keep_bboxes = bboxes[keep]
    print(len(keep_bboxes))
    print(keep_bboxes[:, 4])
    for idx, bbox in enumerate(keep_bboxes):
        _single_bbox = keep_bboxes[idx]
        point1 = (int(_single_bbox[0]), int(_single_bbox[1]))
        point2 = (int(_single_bbox[2]), int(_single_bbox[3]))
        cv2.rectangle(img1, point1, point2, color=[255, 255, 0], thickness=2)
    cv2.imwrite('after.png', img1)
