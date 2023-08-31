"""
pyuic6 -o AppMainWindow/ui_form.py "path/to/file.ui"
"""
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from .ui_form import Ui_MainWindow

import face_recognition as fr
import cv2 as cv
import numpy as np

import os.path as path
import os


def cvMatToQImage(inMat: cv.Mat) -> QImage:
    height, width, channel = inMat.shape
    bytesPerLine = 3 * width
    qImg = QImage(inMat.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    return qImg.rgbSwapped()


def cvMatToQPixmap(inMat: cv.Mat) -> QPixmap:
    return QPixmap.fromImage(cvMatToQImage(inMat))


class AppMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)

        self.cam = cv.VideoCapture(0)
        self.timer = QTimer(self)
        self.maxVideo = QLabel()

        self.known_face_encodings = []
        self.known_face_names = []
        self.lastFrame = None

        self.setupUi(self)
        self._init()

    def __del__(self) -> None:
        self.cam.release()

    def _init(self) -> None:
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.maxVideo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.maxVideo.resize(600, 500)
        self.timer.timeout.connect(self.readCamera)
        self.videoLabel.mouseDoubleClickEvent = self.videoLabel_doubleClicked
        self.maxVideo.mouseDoubleClickEvent = lambda ev: self.maxVideo.hide()
        self.loadData("data")
        self.timer.start(2)

    def _resize(self, img: cv.Mat, screenSize: tuple[int, int]) -> cv.Mat:
        hi, wi = img.shape[:2]
        ws, hs = screenSize
        ri, rs = wi / hi, ws / hs

        wn = int(wi * hs / hi) if (rs > ri) else ws
        hn = hs if (rs > ri) else int(hi * ws / wi)
        wn, hn = max(wn, 1), max(hn, 1)

        if (wn * hn) < (wi * hi):
            return cv.resize(img, (wn, hn), interpolation=cv.INTER_AREA)
        else:
            return cv.resize(img, (wn, hn), interpolation=cv.INTER_LINEAR)

    def loadData(self, dataDir: str) -> None:
        for name in os.listdir(dataDir):
            imgDir = path.join(dataDir, name)
            for file in os.listdir(imgDir):
                image = fr.load_image_file(path.join(imgDir, file))
                faceEncs = fr.face_encodings(image)
                if faceEncs:
                    self.known_face_encodings.append(faceEncs[0])
                    self.known_face_names.append(name)
        print(str(self.known_face_encodings[0]))

    def readCamera(self) -> None:
        ret, frame = self.cam.read()
        if not ret:
            return
        self.lastFrame = frame.copy()

        small_frame = cv.resize(
            frame, (0, 0), fx=0.3, fy=0.3, interpolation=cv.INTER_AREA
        )
        rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"

            face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            best_index = np.argmin(face_distances)

            if face_distances[best_index] < 0.4:
                name = self.known_face_names[best_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right = int(top * 3.3), int(right * 3.3)
            bottom, left = int(bottom * 3.3), int(left * 3.3)

            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(
                frame, (left, bottom - 40), (right, bottom), (0, 0, 255), cv.FILLED
            )

            font = cv.FONT_HERSHEY_COMPLEX
            cv.putText(
                frame, name, (left + 8, bottom - 8), font, 1.5, (255, 255, 255), 2
            )

        if self.maxVideo.isHidden():
            frame = self._resize(
                frame, (self.videoLabel.width(), self.videoLabel.height())
            )
            self.videoLabel.setPixmap(cvMatToQPixmap(frame))
        else:
            frame = self._resize(frame, (self.maxVideo.width(), self.maxVideo.height()))
            self.maxVideo.setPixmap(cvMatToQPixmap(frame))

    def videoLabel_doubleClicked(self, ev: QMouseEvent) -> None:
        if self.maxVideo.isHidden():
            self.maxVideo.showMaximized()
        else:
            self.maxVideo.hide()


"""from mysql.connector import connect

db = connect(
    host="localhost",
    user="root",
    password="abcd1234",
)"""
