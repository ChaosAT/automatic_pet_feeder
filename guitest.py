from PyQt5.QtWidgets import QApplication, QWidget, \
    QPushButton, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap, QImage

import sys
import struct
import cv2
import socket
import time
import os
import numpy as np
import threading
import config as cfg

class Client(object):
    def __init__(self):
        self.SERVER_IP = cfg.SERVER_IP
        self.SERVER_PORT = cfg.SERVER_PORT

        self.socket_tcp = self.tcp_init()
        self.app, self.window = self.ui_init()

        self.info_size = 12
        self.video_state = b'END'

    def tcp_init(self):
        socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.SERVER_IP, self.SERVER_PORT)
        try:
            socket_tcp.connect(server_address)
            print("Connected:", server_address)
        except Exception:
            print("Connected failed.")

        data = socket_tcp.recv(512)
        print(data)
        return socket_tcp

    def process(self, cmd):
        self.socket_tcp.sendall(cmd)

    def reset_button_clicked(self):
        self.process(b'RESET')
        self.label1.setPixmap(QPixmap("ui.png"))

    def img_recv(self):
        info_data = self.socket_tcp.recv(self.info_size)
        info_data = np.frombuffer(info_data, np.int32)
        data_len = info_data[0]
        img_size = (info_data[1], info_data[2])
        buffer_size = 512
        img_data = b""

        while data_len > 0:
            temp_data = self.socket_tcp.recv(buffer_size)
            img_data += temp_data
            data_len -= len(temp_data)
            if buffer_size > data_len:
                buffer_size = data_len
        img_data = np.frombuffer(img_data, np.uint8)
        img_data = np.reshape(img_data, (img_size[0], img_size[1], 3))
        return img_data

    def getimage_button_clicked(self):
        self.process(b'GET_IMAGE')
        img_data = self.img_recv()
        img_data = self.img_cvt(img_data)
        pix_data = QPixmap(img_data)
        self.label1.setPixmap(pix_data)

    def img_cvt(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qimg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        return qimg

    def getvideo_button_clicked(self):
        th = threading.Thread(target=self.play_video)
        th.start()

    def play_video(self):
        self.process(b'GET_VIDEO')

        self.video_state = b'RUN'
        while self.video_state != b'END':
            img_data = self.img_recv()
            img_data = self.img_cvt(img_data)
            pix_data = QPixmap(img_data)
            self.label1.setPixmap(pix_data)
            time.sleep(0.01)
            self.socket_tcp.send(self.video_state)
        self.label1.setPixmap(QPixmap("ui.png"))

    def endvideo_button_clicked(self):
        self.video_state = b'END'

    def gettimerecored_button_clicked(self):
        self.process(b'GET_TIME_RECORD')
        result_text = "Time Record: \n\n"
        cat_time = self.socket_tcp.recv(512)
        cat_time = cat_time.decode('gbk')
        dog_time = self.socket_tcp.recv(512)
        dog_time = dog_time.decode('gbk')
        result_text += cat_time
        result_text += '\n'
        result_text += dog_time
        self.label2.setText(result_text)

    def getdetectresult_button_clicked(self):
        self.process(b'GET_DETECT_RECORD')
        detect_result_num = self.socket_tcp.recv(512)
        detect_result_num = np.frombuffer(detect_result_num, np.int32)[0]
        result_text = "Receive {} detect results.".format(detect_result_num)
        result_text += '\n'
        for i in range(detect_result_num):
            img_name = self.socket_tcp.recv(40)
            img_name = img_name.decode('gbk')
            result_text += img_name
            result_text += '\n'
            img = self.img_recv()
            img_name = os.path.join(os.getcwd(), img_name)
            cv2.imwrite(img_name, img)

        self.label2.setText(result_text)
    def exit_button_clicked(self):
        self.process(b'EXIT')
        self.socket_tcp.close()
        sys.exit(0)

    def ui_init(self):
        app = QApplication([])
        window = QWidget()
        layout = QGridLayout()
        self.label1 = QLabel()
        self.label2 = QLabel()

        self.label1.setFixedSize(500, 500)
        self.label1.setPixmap(QPixmap("ui.png"))
        self.label1.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(300,300,300,120);}")

        self.label2.setFixedSize(350, 500)
        self.label2.setText("Record region.")
        self.label2.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(300,300,300,120);}")

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_button_clicked)
        getimage_button = QPushButton("Get image")
        getimage_button.clicked.connect(self.getimage_button_clicked)
        getvideo_button = QPushButton("Get video")
        getvideo_button.clicked.connect(self.getvideo_button_clicked)
        endvideo_button = QPushButton("End video")
        endvideo_button.clicked.connect(self.endvideo_button_clicked)
        getrecord_button = QPushButton("Get time record")
        getrecord_button.clicked.connect(self.gettimerecored_button_clicked)
        getdetectrecord_button = QPushButton("Get detect record")
        getdetectrecord_button.clicked.connect(self.getdetectresult_button_clicked)
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.exit_button_clicked)

        layout.addWidget(self.label1, 0, 0)
        layout.addWidget(self.label2, 0, 1)
        layout.addWidget(reset_button, 1, 0)
        layout.addWidget(getimage_button, 2, 0)
        layout.addWidget(getvideo_button, 3, 0)
        layout.addWidget(endvideo_button, 4, 0)
        layout.addWidget(getrecord_button, 1, 1)
        layout.addWidget(getdetectrecord_button, 2, 1)
        layout.addWidget(exit_button, 3, 1)


        window.setLayout(layout)
        window.setWindowTitle("Robot control")
        app.setStyle('Fusion')
        return app, window

    def Run(self):
        self.window.show()
        self.app.exec_()


client = Client()
client.Run()
