"""
pyuic6 -o AppMainWindow/ui_mainwindow.py "path/to/file.ui"
pyuic6 -o AppMainWindow/ui_addwindow.py "path/to/file.ui"
pyuic6 -o AppMainWindow/ui_deletewindow.py "path/to/file.ui"
"""
from PyQt6.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent
from PyQt6.QtCore import QTimer, Qt
from .ui_mainwindow import Ui_MainWindow
from .ui_addwindow import Ui_Widget

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


class AddWindow(QWidget, Ui_Widget):
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


class AppMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)

        self.cam = cv.VideoCapture(0)
        self.timer = QTimer(self)
        self.maxVideo = QLabel()
        self.addWindow = AddWindow()

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

        self.addWindow.browseChoice.clicked.connect(self.browseChoice_clicked)
        self.addWindow.screenshotChoice.clicked.connect(self.screenshotChoice_clicked)
        self.addWindow.platStopBtn.clicked.connect(self.platStopBtn_clicked)
        self.addWindow.proceedBtn.clicked.connect(self.proceedBtn_clicked)
        self.addWindow.okBtn.clicked.connect(self.okBtn_clicked)
        self.addWindow.selectBtn.clicked.connect(self.selectBtn_clicked)
        self.addWindow.browseProceedBtn.clicked.connect(self.browseProceedBtn_clicked)

        self.videoLabel.mouseDoubleClickEvent = self.videoLabel_doubleClicked
        self.maxVideo.mouseDoubleClickEvent = lambda ev: self.maxVideo.hide()
        self.addWindow.hideEvent = lambda ev: self.timer.start(2)

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
        self.cr.execute("SELECT name, encoding FROM known_faces")
        for name, encoding in self.cr.fetchall():
            self.known_face_names.append(name)
            self.known_face_encodings.append(np.frombuffer(encoding))

    def readCamera(self) -> None:
        ret, self.frame = self.cam.read()
        if not ret:
            return

        if not self.addWindow.isHidden():
            screenSize = (
                self.addWindow.imageLabel.width(),
                self.addWindow.imageLabel.height(),
            )
            videoLabel = self.addWindow.imageLabel
        else:
            screenSize = (
                self.videoLabel.width(),
                self.videoLabel.height(),
            )
            videoLabel = self.videoLabel

        self.frame = self._resize(self.frame, screenSize)

        small_frame = cv.resize(
            self.frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA
        )
        small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        self.face_locations = fr.face_locations(small_frame)
        self.face_encodings = fr.face_encodings(small_frame, self.face_locations)

        if self.known_face_names:
            self.face_names = []

            for face_encoding in self.face_encodings:
                name = "Unknown"

                face_distances = fr.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_index = np.argmin(face_distances)

                if face_distances[best_index] < 0.4:
                    name = self.known_face_names[best_index]
                self.face_names.append(name)
        else:
            self.face_names = ["Unknown" for _ in range(len(self.face_locations))]

        for (top, right, bottom, left), name in zip(
            self.face_locations, self.face_names
        ):
            top, right = top * 2, right * 2
            bottom, left = bottom * 2, left * 2

            cv.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(
                self.frame, (left, bottom - 20), (right, bottom), (0, 0, 255), cv.FILLED
            )

            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(
                self.frame, name, (left + 5, bottom - 5), font, 1, (255, 255, 255), 1
            )

        if not self.maxVideo.isHidden():
            self.frame = self._resize(
                self.frame, (self.maxVideo.width(), self.maxVideo.height())
            )
            videoLabel = self.maxVideo
        videoLabel.setPixmap(cvMatToQPixmap(self.frame))

    def addBtn_clicked(self) -> None:
        if (not self.addWindow.screenshotChoice.isChecked()) or (
            self.addWindow.platStopBtn.text() == "Play"
        ):
            self.timer.stop()
        self.addWindow.showNormal()

    def browseChoice_clicked(self) -> None:
        self.timer.stop()
        self.addWindow.browseGroup.setEnabled(True)
        self.addWindow.screenshotGroup.setEnabled(False)

    def screenshotChoice_clicked(self) -> None:
        if self.addWindow.platStopBtn.text() == "Stop":
            self.timer.start(2)
        self.addWindow.screenshotGroup.setEnabled(True)
        self.addWindow.browseGroup.setEnabled(False)

    def platStopBtn_clicked(self) -> None:
        if self.addWindow.platStopBtn.text() == "Stop":
            self.timer.stop()
            self.addWindow.platStopBtn.setText("Play")
            self.addWindow.proceedBtn.setEnabled(True)
        else:
            self.timer.start(2)
            self.addWindow.platStopBtn.setText("Stop")
            self.addWindow.proceedBtn.setEnabled(False)

    def proceedBtn_clicked(self) -> None:
        self.uknown_faces.clear()
        for (top, right, bottom, left), name, encoding in zip(
            self.face_locations, self.face_names, self.face_encodings
        ):
            if name != "Unknown":
                continue
            top, right = top * 2, right * 2
            bottom, left = bottom * 2, left * 2

            face = QLineEdit(name, self.addWindow.imageLabel)
            face.encoding = encoding
            face.setStyleSheet("background: red; color: white;")
            face.move(left, bottom + 12)
            face.show()
            self.uknown_faces.append(face)

        self.addWindow.okBtn.setEnabled(True)

    def selectBtn_clicked(self) -> None:
        pass

    def browseProceedBtn_clicked(self) -> None:
        pass

    def okBtn_clicked(self) -> None:
        sql = "INSERT INTO known_faces (name, encoding) VALUES (%s, %s)"
        faces = [(face.text(), face.encoding) for face in self.uknown_faces]
        val = [(name, encoding.tobytes()) for (name, encoding) in faces]

        for name, encoding in faces:
            self.known_face_names.append(name)
            self.known_face_encodings.append(encoding)

        self.cr.executemany(sql, val)
        self.db.commit()
        self.addWindow.okBtn.setEnabled(False)
        self.addWindow.proceedBtn.setEnabled(False)

        for face in self.uknown_faces:
            face.close()
        self.addWindow.imageLabel.clear()

    def videoLabel_doubleClicked(self, ev: QMouseEvent) -> None:
        if self.maxVideo.isHidden():
            self.maxVideo.showMaximized()
        else:
            self.maxVideo.hide()
