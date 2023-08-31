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

        self.setupUi(self)
        self._init()

    def __del__(self) -> None:
        self.cam.release()

    def _init(self) -> None:
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer.timeout.connect(self.readCamera)
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

    def readCamera(self) -> None:
        ret, frame = self.cam.read()
        if not ret:
            return
        frame = self._resize(frame, (self.videoLabel.width(), self.videoLabel.height()))
        self.videoLabel.setPixmap(cvMatToQPixmap(frame))


"""from mysql.connector import connect

db = connect(
    host="localhost",
    user="root",
    password="abcd1234",
)"""

"""import face_recognition as fr
import cv2 as cv
import numpy as np
import os.path as path
import os

DATA_DIR = "data"

known_face_encodings = []
known_face_names = []

for name in os.listdir(DATA_DIR):
    imgDir = path.join(DATA_DIR, name)
    for file in os.listdir(imgDir):
        image = fr.load_image_file(path.join(imgDir, file))
        faceEncs = fr.face_encodings(image)
        if faceEncs:
            known_face_encodings.append(faceEncs[0])
            known_face_names.append(name)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    frame = cv.resize(frame, (0, 0), fx=0.7, fy=0.7, interpolation=cv.INTER_AREA)

    if process_this_frame:
        small_frame = cv.resize(
            frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA
        )

        rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"

            face_distances = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right = top * 2, right * 2
        bottom, left = bottom * 2, left * 2

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED
        )
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(2) == 27:  # esc
        break

video_capture.release()
cv.destroyAllWindows()"""
