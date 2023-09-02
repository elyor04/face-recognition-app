"""
pyuic6 -o AppMainWindow/ui_mainwindow.py "path/to/file.ui"
pyuic6 -o AppMainWindow/ui_addwindow.py "path/to/file.ui"
pyuic6 -o AppMainWindow/ui_deletewindow.py "path/to/file.ui"
"""
import typing
from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit, QFileDialog
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QHideEvent
from PyQt6.QtCore import QTimer, Qt
from .ui_mainwindow import Ui_MainWindow
from .ui_addwindow import Ui_Widget as Ui_AddWindow
from .ui_deletewindow import Ui_Widget as Ui_DeleteWindow

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


class AddWindow(QWidget, Ui_AddWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self._init()

    def _init(self) -> None:
        self.browseChoice.setChecked(True)
        self.okBtn.setEnabled(False)
        self.proceedBtn.setEnabled(False)
        self.screenshotGroup.setEnabled(False)
        self.browseProceedBtn.setEnabled(False)
        self.cancelBtn.clicked.connect(self.hide)
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)


class DeleteWindow(QWidget, Ui_DeleteWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self._init()

    def _init(self) -> None:
        self.okBtn.setEnabled(False)
        self.cancelBtn.clicked.connect(self.hide)


class AppMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)

        self.cam = cv.VideoCapture(0)
        self.timer = QTimer(self)
        self.maxVideo = QLabel()
        self.addWindow = AddWindow()
        self.delWindow = DeleteWindow()

        self.known_face_encodings = []
        self.known_face_names = []
        self.uknown_faces = []

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.frame = None

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
        self.addBtn.clicked.connect(self.addBtn_clicked)
        self.maxVideo.setStyleSheet("background: rgb(150, 150, 150);")

        self.addWindow.browseChoice.clicked.connect(self.browseChoice_clicked)
        self.addWindow.screenshotChoice.clicked.connect(self.screenshotChoice_clicked)
        self.addWindow.platStopBtn.clicked.connect(self.platStopBtn_clicked)
        self.addWindow.proceedBtn.clicked.connect(self.proceedBtn_clicked)
        self.addWindow.okBtn.clicked.connect(self.okBtn_clicked)
        self.addWindow.openBtn.clicked.connect(self.openBtn_clicked)
        self.addWindow.browseProceedBtn.clicked.connect(self.browseProceedBtn_clicked)

        self.videoLabel.mouseDoubleClickEvent = self.videoLabel_doubleClicked
        self.maxVideo.mouseDoubleClickEvent = lambda ev: self.maxVideo.hide()
        self.addWindow.hideEvent = self.addWindow_hideEvent

        self.createConnection()
        self.loadData()
        self.timer.start(2)

    def addWindow_hideEvent(self, ev: QHideEvent) -> None:
        self.timer.start(2)
        for face in self.uknown_faces:
            face.close()
        self.addWindow.imageLabel.clear()
        self.addWindow.okBtn.setEnabled(False)

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
        self.cr.execute("SELECT name, encoding FROM known_faces")
        for name, encoding in self.cr.fetchall():
            self.known_face_names.append(name)
            self.known_face_encodings.append(np.frombuffer(encoding))

    def _detectFaces(self, img: cv.Mat, scale: float = 0.7) -> tuple[list, list, list]:
        small_frame = cv.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA
        )
        small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        face_locations = fr.face_locations(small_frame)
        face_encodings = fr.face_encodings(small_frame, face_locations)

        if self.known_face_names:
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

        return (face_locations, face_encodings, face_names)

    def _visualize(
        self, img: cv.Mat, face_locations: list, face_names: list, scale: float = 0.7
    ) -> cv.Mat:
        scale = 1 / scale
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right = int(top * scale), int(right * scale)
            bottom, left = int(bottom * scale), int(left * scale)

            cv.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(
                img, (left - 1, bottom - 15), (right, bottom + 5), (0, 0, 255), -1
            )

            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(img, name, (left - 2, bottom), font, 1, (255, 255, 255), 1)
        return img

    def _detectAndVisualizeFaces(
        self, img: cv.Mat, scale: float = 0.7
    ) -> tuple[list, list, list]:
        face_locations, face_encodings, face_names = self._detectFaces(img, scale)
        self._visualize(img, face_locations, face_names, scale)
        return (face_locations, face_encodings, face_names)

    def readCamera(self) -> None:
        ret, frame = self.cam.read()
        if not ret:
            return

        if not self.addWindow.isHidden():
            frame = self._resize(
                frame,
                (self.addWindow.imageLabel.width(), self.addWindow.imageLabel.height()),
            )
            self.frame = frame.copy()
            self._detectAndVisualizeFaces(frame)
            self.addWindow.imageLabel.setPixmap(cvMatToQPixmap(frame))

        elif not self.maxVideo.isHidden():
            frame = self._resize(frame, (self.maxVideo.width(), self.maxVideo.height()))
            scale = (self.videoLabel.width() * self.videoLabel.height()) / (
                self.maxVideo.width() * self.maxVideo.height()
            )
            scale = min(scale * 0.7, 1)
            self._detectAndVisualizeFaces(frame, scale)
            self.maxVideo.setPixmap(cvMatToQPixmap(frame))

        else:
            frame = self._resize(
                frame, (self.videoLabel.width(), self.videoLabel.height())
            )
            self._detectAndVisualizeFaces(frame)
            self.videoLabel.setPixmap(cvMatToQPixmap(frame))

    def addBtn_clicked(self) -> None:
        if (not self.addWindow.screenshotChoice.isChecked()) or (
            self.addWindow.platStopBtn.text() == "Play"
        ):
            self.timer.stop()
        self.addWindow.showNormal()
        self.videoLabel.clear()

    def browseChoice_clicked(self) -> None:
        self.timer.stop()
        self.addWindow.browseGroup.setEnabled(True)
        self.addWindow.screenshotGroup.setEnabled(False)
        for face in self.uknown_faces:
            face.close()
        self.addWindow.imageLabel.clear()

    def screenshotChoice_clicked(self) -> None:
        if self.addWindow.platStopBtn.text() == "Stop":
            self.timer.start(2)
        self.addWindow.screenshotGroup.setEnabled(True)
        self.addWindow.browseGroup.setEnabled(False)
        for face in self.uknown_faces:
            face.close()
        self.addWindow.imageLabel.clear()

    def platStopBtn_clicked(self) -> None:
        if self.addWindow.platStopBtn.text() == "Stop":
            self.timer.stop()
            self.addWindow.platStopBtn.setText("Play")
            self.addWindow.proceedBtn.setEnabled(True)
            (
                self.face_locations,
                self.face_encodings,
                self.face_names,
            ) = self._detectAndVisualizeFaces(self.frame, 1)
            self.addWindow.imageLabel.setPixmap(cvMatToQPixmap(self.frame))
        else:
            self.timer.start(2)
            self.addWindow.platStopBtn.setText("Stop")
            self.addWindow.proceedBtn.setEnabled(False)
            for face in self.uknown_faces:
                face.close()
            self.addWindow.imageLabel.clear()

    def _proceed(self, scale: float = 1.0) -> None:
        scale = 1 / scale
        self.uknown_faces.clear()
        deltaX = (self.addWindow.imageLabel.width() - self.frame.shape[1]) // 2
        deltaY = (self.addWindow.imageLabel.height() - self.frame.shape[0]) // 2

        for (top, right, bottom, left), name, encoding in zip(
            self.face_locations, self.face_names, self.face_encodings
        ):
            if self.known_face_names.count(name) >= 10:
                continue
            top, right = int(top * scale), int(right * scale)
            bottom, left = int(bottom * scale), int(left * scale)

            face = QLineEdit(name, self.addWindow.imageLabel)
            face.encoding = encoding
            face.setStyleSheet("background: red; color: white;")
            face.move(left + deltaX - 2, bottom + deltaY - 15)
            face.setFixedWidth(right - left + 5)
            face.show()
            self.uknown_faces.append(face)

        self.addWindow.okBtn.setEnabled(True)

    def proceedBtn_clicked(self) -> None:
        self._proceed()

    def openBtn_clicked(self) -> None:
        formats = [
            "*.jpeg",
            "*.jpg",
            "*.jpe",
            "*.jp2",
            "*.png",
            "*.bmp",
            "*.dib",
            "*.webp",
        ]
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", filter=f"Image Files ({' '.join(formats)})"
        )

        for face in self.uknown_faces:
            face.close()
        self.addWindow.imageLabel.clear()

        if fileName:
            self.frame = self._resize(
                cv.imread(fileName),
                (self.addWindow.imageLabel.width(), self.addWindow.imageLabel.height()),
            )
            (
                self.face_locations,
                self.face_encodings,
                self.face_names,
            ) = self._detectAndVisualizeFaces(self.frame, 1)

            self.addWindow.browseProceedBtn.setEnabled(True)
            self.addWindow.imageLabel.setPixmap(cvMatToQPixmap(self.frame))

    def browseProceedBtn_clicked(self) -> None:
        self._proceed()

    def okBtn_clicked(self) -> None:
        sql = "INSERT INTO known_faces (name, encoding) VALUES (%s, %s)"
        faces = [
            (face.text(), face.encoding)
            for face in self.uknown_faces
            if (face.text() != "Unknown")
        ]
        val = [(name, encoding.tobytes()) for (name, encoding) in faces]

        for name, encoding in faces:
            self.known_face_names.append(name)
            self.known_face_encodings.append(encoding)

        self.cr.executemany(sql, val)
        self.db.commit()

        self.addWindow.okBtn.setEnabled(False)
        self.addWindow.proceedBtn.setEnabled(False)
        self.addWindow.browseProceedBtn.setEnabled(False)

        for face in self.uknown_faces:
            face.close()
        self.addWindow.imageLabel.clear()

    def videoLabel_doubleClicked(self, ev: QMouseEvent) -> None:
        if self.maxVideo.isHidden():
            self.maxVideo.showMaximized()
            self.videoLabel.clear()
        else:
            self.maxVideo.hide()
