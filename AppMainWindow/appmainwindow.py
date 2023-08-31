"""
pyuic6 -o AppMainWindow/ui_form.py "path/to/file.ui"
"""
from PyQt6.QtWidgets import QMainWindow, QWidget
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
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
        self.known_face_encodings = []
        self.known_face_names = []

        self.setupUi(self)
        self._init()

    def __del__(self) -> None:
        self.cam.release()

    def _init(self) -> None:
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer.timeout.connect(self.readCamera)
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

    def readCamera(self) -> None:
        ret, frame = self.cam.read()
        if not ret:
            return

        small_frame = cv.resize(
            frame, (0, 0), fx=0.4, fy=0.4, interpolation=cv.INTER_AREA
        )
        rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"

            face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.4:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right = int(top * 2.5), int(right * 2.5)
            bottom, left = int(bottom * 2.5), int(left * 2.5)

            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(
                frame, (left, bottom - 40), (right, bottom), (0, 0, 255), cv.FILLED
            )

            font = cv.FONT_HERSHEY_COMPLEX
            cv.putText(
                frame, name, (left + 8, bottom - 8), font, 1.5, (255, 255, 255), 2
            )

        frame = self._resize(frame, (self.videoLabel.width(), self.videoLabel.height()))
        self.videoLabel.setPixmap(cvMatToQPixmap(frame))


"""from mysql.connector import connect

db = connect(
    host="localhost",
    user="root",
    password="abcd1234",
)"""
