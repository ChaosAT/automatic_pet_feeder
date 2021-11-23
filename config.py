"""
    需要的各种参数
"""
#通信地址
SERVER_IP = "192.168.1.7"
CLIENT_IP = "192.168.1.4"

SERVER_PORT = 10000
CLIENT_PORT = 10001

#超声波所用引脚
GPIO_SOUND_TRIGGER = 2
GPIO_SOUND_ECHO = 3

#舵机所用引脚
GPIO_MOTOR_CAT = 4
GPIO_MOTOR_DOG = 5

#超声波距离阈值，设为10cm
SOUND_TRESH = 10

#相机帧率
FRAME = 20

#相机获取的图像大小
IMAGE_SIZE = 608

#tflite模型位置
DETECT_MODEL_PATH = "mbv3yolo.tflite"

#用于检测的图像大小
DETECT_SIZE = 224

#用于检测的类别list
CLASS_NAME = ['cat', 'dog', 'person']

#用于检测的参数
MAX_BOXNUM = 2
GRID_NUM = 7
GRID_SIZE = IMAGE_SIZE//GRID_NUM

#用于检测食盆余量的图像大小
FOOD_CHECK_SIZE = 32

#最小喂食间隔时间（秒）
FEED_TIME = 60
