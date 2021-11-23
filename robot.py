import cv2
import time
import config as cfg
import numpy as np
import RPi.GPIO as GPIO
import picamera as pcam
from detect import Detector

class Robot(object):

    def __init__(self):

        self.cat = 0
        self.dog = 1

        self.gpio_trigger = -1
        self.gpio_echo = -1
        self.gpio_motor_cat = -1
        self.gpio_motor_dog = -1
        self.pwm_cat = None
        self.pwm_dog = None
        self.gpio_init()
        print("GPIO init success.")

        self.sound_thresh = cfg.SOUND_TRESH
        self.init_dist = self.get_dist()

        self.class_name = cfg.CLASS_NAME
        self.detector = Detector()
        self.img_size = cfg.IMAGE_SIZE
        self.camera = pcam.PiCamera()
        self.camera.resolution = (self.img_size, self.img_size)
        self.camera.framerate = cfg.FRAME

        self.capture_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        self.FeedTime = {}
        self.FeedTime['cat'] = []
        self.FeedTime['cat'].append(time.time())

        self.FeedTime['dog'] = []
        self.FeedTime['dog'].append(time.time())
        self.FeedTime['time'] = cfg.FEED_TIME

    def time_transform(self, input_time):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(input_time))

    def gpio_init(self):
        GPIO.setmode(GPIO.BCM)

        #初始化超声波gpio
        self.gpio_trigger = cfg.GPIO_SOUND_TRIGGER
        self.gpio_echo = cfg.GPIO_SOUND_ECHO
        GPIO.setup(self.gpio_trigger, GPIO.OUT)
        GPIO.setup(self.gpio_echo, GPIO.IN)

        #初始化舵机gpio，设置pwm
        self.gpio_motor_cat = cfg.GPIO_MOTOR_CAT
        self.gpio_motor_dog = cfg.GPIO_MOTOR_DOG

        GPIO.setup(self.gpio_motor_cat, GPIO.OUT, initial=False)
        GPIO.setup(self.gpio_motor_dog, GPIO.OUT, initial=False)
        self.pwm_cat = GPIO.PWM(self.gpio_motor_cat, 50)  # 50HZ
        self.pwm_dog = GPIO.PWM(self.gpio_motor_dog, 50)  # 50HZ
        self.pwm_cat.start(0)
        self.pwm_dog.start(0)
        time.sleep(2)
        self.motor_reset(self.pwm_cat)
        self.motor_reset(self.pwm_dog)


    def motor_reset(self, pwm):
        pwm.ChangeDutyCycle(7.5)  # 设置转动角度
        time.sleep(0.1)
        pwm.ChangeDutyCycle(0)  # 归零信号
        time.sleep(2)

    def get_dist(self):
        GPIO.output(self.gpio_trigger, True)
        # 持续 10 us
        time.sleep(0.00001)
        GPIO.output(self.gpio_trigger, False)
        start_time = time.time()
        stop_time = time.time()

        # 记录发送超声波的时刻1
        while GPIO.input(self.gpio_echo) == 0:
            start_time = time.time()
        # 记录接收到返回超声波的时刻2
        while GPIO.input(self.gpio_echo) == 1:
            stop_time = time.time()
        # 计算超声波的往返时间 = 时刻2 - 时刻1
        time_elapsed = stop_time - start_time
        # 声波的速度为 343m/s， 转化为 34300cm/s。
        dist = (time_elapsed * 34300) / 2
        return dist

    def CheckMovement(self):
        #使用超声波传感器检测位移
        print("Starting supersonic detect...")
        if np.abs(self.get_dist()-self.init_dist) > self.sound_thresh:
            print("!!! Movements detected. !!!")
            return True
        else:
            print("No movements.")
            return False

    def get_image(self, for_video=False):
        """
            1.从摄像头里抓取一张拍到2个食盆和场景的高分辨率图像.
            2.将抓取到的图像分割成3个部分，猫食盆、狗食盆和场景.
            3.2个食盆的图像转换32X32的灰度图，场景的图像为448X448的RGB图像.

            :return:
        """
        if for_video:
            self.camera.capture(self.capture_img, format='rgb', use_video_port=True)
            return self.capture_img

        self.camera.start_preview()
        time.sleep(0.05)
        self.camera.capture(self.capture_img, format='rgb', use_video_port=False)
        scene = self.capture_img[:448, :self.img_size, :]
        pots = self.capture_img[450:, :self.img_size, :]
        pots = cv2.cvtColor(pots, cv2.COLOR_RGB2GRAY)
        catpot = pots[:, :300]
        dogpot = pots[:, 300:]
        return scene, catpot, dogpot, self.capture_img

    def motor_run(self, pwm):
        #打开阀门放粮
        pwm.ChangeDutyCycle(2.5)
        time.sleep(0.15)
        pwm.ChangeDutyCycle(0)
        time.sleep(3)

        #关闭阀门
        pwm.ChangeDutyCycle(7.5)
        time.sleep(0.15)
        pwm.ChangeDutyCycle(0)
        time.sleep(3)
        pass

    def ResupplyFood(self, pot):
        print("Start to resupplyfood...")
        print("Resupplying:", self.class_name[pot])
        if pot == self.cat:
            self.motor_run(self.pwm_cat)
        if pot == self.dog:
            self.motor_run(self.pwm_dog)
        print("Resupply ends.")

    def CheckFood(self):
        """
        检测猫狗粮盆里是否有余量，有余量返回False，无余量需要补充返回True
        :return:
        """
        print("Check if there still some food in pots...")
        _, catpot, dogpot, _ = self.get_image()
        cat_pot_status = self.detector.pot_classifier(catpot)  #
        dog_pot_status = self.detector.pot_classifier(dogpot)  #

        return cat_pot_status, dog_pot_status

    def CheckTime(self):
        """
        查看与上次喂食是否间隔了6小时，小于6小时返回False，大于6小时需喂食返回True
        :return:
        """
        print("Check if it's time to feed...")
        time_now = time.time()
        cat_time = time_now - self.FeedTime['cat'][-1]
        dog_time = time_now - self.FeedTime['dog'][-1]
        cat_time_status = cat_time > self.FeedTime['time']
        dog_time_status = dog_time > self.FeedTime['time']
        print("Cat time:", cat_time, cat_time_status)
        print("Dog time:", dog_time, dog_time_status)
        return cat_time_status, dog_time_status

    def DetectObject(self):
        scene, _, _, _ = self.get_image()
        result_image, result_score, result_class = self.detector.detect_image(scene)
        time_str = self.time_transform(time.time())
        time_str = time_str.replace(' ', '_')
        time_str = time_str.replace(':', '-')
        cv2.imwrite("detect_result_at_{}.jpg".format(time_str), result_image)

        return result_score, result_class

    def CheckObject(self):
        """
            返回一个包含检测结果的list, 猫为0、狗为1、人为2，
            无结果时返回空list
        :return:
        """
        print("Starting object detection...")
        result = []
        result_score, result_class = self.DetectObject()
        if np.sum(result_class) == -1:
            print("CheckObject: No Object.")
        else:
            for i in range(result_class.shape[0]):
                if result_class[i] == 7: result.append(0)
                if result_class[i] == 11: result.append(1)
                if result_class[i] == 19: result.append(2)
            for i in range(len(result)):
                print("Detection results:", self.class_name[result[i]])
        print("Detection ends.")
        return result

    def SaveTimeRecord(self, obj_class):
        self.FeedTime[obj_class].append(time.time())

    def GetTimeRecord(self):
        feedtime = {}
        feedtime['cat'] = []
        feedtime['dog'] = []

        for i in range(len(self.FeedTime['cat'])):
            feedtime['cat'].append(self.time_transform(self.FeedTime['cat'][i]))
        for i in range(len(self.FeedTime['dog'])):
            feedtime['dog'].append(self.time_transform(self.FeedTime['dog'][i]))

        return feedtime


    def RaiseError(self):
        print("Error.")
