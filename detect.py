import cv2
import time
import config as cfg
import numpy as np
from evaluate import DetectParser
import tensorflow as tf

class Detector():
    def __init__(self):
        self.model = tf.lite.Interpreter(model_path=cfg.DETECT_MODEL_PATH)
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.model.allocate_tensors()
        self.img_size = cfg.DETECT_SIZE
        self.parser = DetectParser(batch_size=1)
        self.time_cost = []

    def pot_classifier(self, img):
        """
        一个简单的分类器，分辨图片中的食盆有无余量.
        :param img:
        :return:Boolean
        """
        img_mean = np.mean(img)
        if img_mean < 170:  # 图像偏暗可认为有余量
            return True
        else:
            return False

    def detect_image(self, img):
        st = time.time()
        img = cv2.resize(img, (self.img_size, self.img_size))  # 可以删掉
        result_image = img
        img_data = np.copy(img)
        img_data = img_data.astype(np.float32) / 127.5 - 1.0
        img_data = np.expand_dims(img_data, 0)

        self.model.set_tensor(self.input_details[0]['index'], img_data)
        self.model.invoke()
        pred = self.model.get_tensor(self.output_details[0]['index'])

        mid = time.time()
        result_image, result_bbox, result_score, result_class = self.parser.parse_pred(pred, result_image,
                                              class_num=20, iou_thresh=0.4, score_thresh=0.3)
        ed = time.time()
        print("Detect time:{:.2f}sec".format(mid - st), "Process bbox time:{:.2f}sec".format(ed - mid))
        return result_image, result_score, result_class
