import tensorflow as tf
import numpy as np
import cv2
import config as cfg


# 解析出所有bbox，分为NX98X4的坐标，NX98X1的置信度，NX98X1的类别
class DetectParser(object):

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.grid_num = cfg.GRID_NUM
        self.max_boxnum = cfg.MAX_BOXNUM
        self.box_num = self.grid_num * self.grid_num * self.max_boxnum
        self.grid_offset = self.grid_offset_init()

    def grid_offset_init(self):
        grid_y, grid_x = tf.meshgrid(tf.range(cfg.GRID_NUM), tf.range(cfg.GRID_NUM))
        grid_offset = tf.stack([grid_y, grid_x], axis=2)  # 7X7X2
        grid_offset = tf.expand_dims(grid_offset, 0)
        grid_offset = tf.tile(grid_offset, [self.batch_size, 1, 1, 1])  # NX7X7X2
        grid_offset = tf.expand_dims(grid_offset, -2)
        grid_offset = tf.tile(grid_offset, [1, 1, 1, 2, 1])
        grid_offset = tf.cast(grid_offset, dtype=tf.float32)
        return grid_offset

    def draw_box(self, img, box):
        pt1 = (box[0] - box[2] // 2, box[1] - box[3] // 2)
        pt2 = (box[0] + box[2] // 2, box[1] + box[3] // 2)
        draw1 = cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=1)
        return draw1

    def calcu_iou(self, y_pred_bboxes, y_true_bboxes):
        minx = tf.maximum(y_pred_bboxes[..., 0] - y_pred_bboxes[..., 2] / 2,
                          y_true_bboxes[..., 0] - y_true_bboxes[..., 2] / 2)
        maxx = tf.minimum(y_pred_bboxes[..., 0] + y_pred_bboxes[..., 2] / 2,
                          y_true_bboxes[..., 0] + y_true_bboxes[..., 2] / 2)
        isx = tf.maximum(maxx - minx, 0.0)

        miny = tf.maximum(y_pred_bboxes[..., 1] - y_pred_bboxes[..., 3] / 2,
                          y_true_bboxes[..., 1] - y_true_bboxes[..., 3] / 2)
        maxy = tf.minimum(y_pred_bboxes[..., 1] + y_pred_bboxes[..., 3] / 2,
                          y_true_bboxes[..., 1] + y_true_bboxes[..., 3] / 2)
        isy = tf.maximum(maxy - miny, 0.0)

        intersection_s = isx * isy  # NX7X7X2X1
        y_pred_bboxes_s = y_pred_bboxes[..., 2] * y_pred_bboxes[..., 3]  # NX7X7X2X1
        y_true_bboxes_s = y_true_bboxes[..., 2] * y_true_bboxes[..., 3]

        union_s = tf.maximum(y_true_bboxes_s + y_pred_bboxes_s - intersection_s, 1e-10)
        iou = intersection_s / union_s
        # 可能小于0，可能大于1，输出限制在0～1
        return tf.clip_by_value(iou, 0.0, 1.0).numpy()  # NX7X7X2X1

    def cpu_nms(self, box, score, bbox_class, iou_thresh=0.4, score_thresh=0.3):
        # box NX98X4
        # conf NX98
        box_num = np.shape(box)[0]
        box = box[0]
        score = score[0]
        bbox_class = bbox_class[0]
        order_index = np.argsort(score)[::-1]
        #  NXn
        score = score[order_index]
        box = box[order_index]
        bbox_class = bbox_class[order_index]

        max_idx = box_num
        for i in range(score.shape[0]):
            if score[i] < score_thresh:
                max_idx = i
                break
        result_idx = []
        for i in range(max_idx):
            if score[i] > 0:
                result_idx.append(i)
            for j in range(i + 1, max_idx):
                if self.calcu_iou(box[i], box[j]) > iou_thresh:
                    score[j] = 0
        if len(result_idx) == 0:
            return False, box, score, bbox_class
        box = box[result_idx]
        score = score[result_idx]
        bbox_class = bbox_class[result_idx]
        return True, box, score, bbox_class

    def parse_pred(self, pred, result_image, class_num=20, iou_thresh=0.4, score_thresh=0.3):

        box_num = self.box_num
        batch_size = self.batch_size

        y_pred_bboxes_offset = tf.reshape(pred[..., 0:8],
                                          [-1, self.grid_num, self.grid_num, self.max_boxnum, 4])  # NX7X7X2X4

        y_pred_bboxes_normalize = tf.stack([(tf.sigmoid(y_pred_bboxes_offset[..., 0]) + self.grid_offset[..., 0]),
                                            (tf.sigmoid(y_pred_bboxes_offset[..., 1]) + self.grid_offset[..., 1]),
                                            tf.square(y_pred_bboxes_offset[..., 2]) * self.grid_num,
                                            tf.square(y_pred_bboxes_offset[..., 3]) * self.grid_num], axis=4)

        pred_coor = tf.reshape(y_pred_bboxes_normalize, [-1, box_num, 4])  # NX98X4

        pred_conf = tf.reshape(tf.sigmoid(pred[..., 8:10]),
                               [batch_size, self.grid_num, self.grid_num, self.max_boxnum, 1])  # NX7X7X2X1

        pred_conf = tf.reshape(pred_conf, [batch_size, box_num, 1])  # NX98X1

        pred_class_score = tf.expand_dims(tf.sigmoid(pred[..., 10:]), -2)  # NX7X7X1X20
        pred_class_score = tf.tile(pred_class_score, [1, 1, 1, 2, 1])  # NX7X7X2X20

        pred_class_score = tf.reshape(pred_class_score, [batch_size, box_num, class_num])  # NX98X20
        max_class_score = tf.reduce_max(pred_class_score, axis=-1)  # NX98
        pred_class = tf.argmax(pred_class_score, axis=2)  # NX98
        bbox_score = max_class_score * pred_conf[..., 0]

        pred_coor = pred_coor.numpy()  # NX98X4
        pred_score = bbox_score.numpy()  # NX98
        pred_class = pred_class.numpy()  # NX98

        has_obj, result_bbox, result_score, result_class = self.cpu_nms(pred_coor, pred_score, pred_class,
                                                                        iou_thresh=iou_thresh, score_thresh=score_thresh)
        if not has_obj:
            return result_image, -1, -1, -1

        result_bbox = (result_bbox * 224.0 / 7.0).astype(np.int32)
        for i in range(result_score.shape[0]):
            result_image = self.draw_box(result_image, result_bbox[i])

        return result_image, result_bbox, result_score, result_class