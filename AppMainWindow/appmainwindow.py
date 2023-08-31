"""
pyuic6 -o AppMainWindow/ui_mainwindow.py "path/to/file.ui"
"""
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtCore import QTimer, Qt
from .ui_mainwindow import Ui_MainWindow

from mysql.connector import connect
import face_recognition as fr
import numpy as np
import cv2 as cv


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

        self.db = None
        self.cr = None

        self.setupUi(self)
        self._init()

    def __del__(self) -> None:
        self.cam.release()
        self.db.close()

    def _init(self) -> None:
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.maxVideo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.maxVideo.resize(600, 500)
        self.timer.timeout.connect(self.readCamera)
        self.videoLabel.mouseDoubleClickEvent = self.videoLabel_doubleClicked
        self.maxVideo.mouseDoubleClickEvent = lambda ev: self.maxVideo.hide()
        self.createConnection()
        self.loadData()
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

    def createConnection(self) -> None:
        self.db = connect(
            host="localhost",
            user="root",
            password="abcd1234",
        )
        self.cr = self.db.cursor()

        self.cr.execute("CREATE DATABASE IF NOT EXISTS face_recognition_app")
        self.db.connect(database="face_recognition_app")

        self.cr.execute(
            "CREATE TABLE IF NOT EXISTS known_faces (id INT AUTO_INCREMENT PRIMARY KEY, name TEXT, encoding BLOB)"
        )

    def loadData(self) -> None:
        """img = fr.load_image_file("data/Elyor/elyor.jpg")
        encoding = fr.face_encodings(img)[0].tobytes()
        name = "Elyor"
        sql = "INSERT INTO known_faces (name, encoding) VALUES (%s, %s)"
        self.cr.execute(sql, (name, encoding))
        self.db.commit()"""

        self.cr.execute("SELECT name, encoding FROM known_faces")
        for name, encoding in self.cr.fetchall():
            self.known_face_encodings.append(np.frombuffer(encoding))
            self.known_face_names.append(name)

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

        if self.known_face_names:
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                name = "Unknown"

                face_distances = fr.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_index = np.argmin(face_distances)

                if face_distances[best_index] < 0.4:
                    name = self.known_face_names[best_index]
                face_names.append(name)
        else:
            face_names = ["Unknown" for _ in range(len(face_locations))]

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
