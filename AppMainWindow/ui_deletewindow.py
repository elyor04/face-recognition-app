# Form implementation generated from reading ui file '/Users/elyor/Documents/qt-projects/delete-window/widget.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(460, 478)
        Widget.setStyleSheet("font: 14pt;")
        self.deleteTable = QtWidgets.QTableWidget(parent=Widget)
        self.deleteTable.setGeometry(QtCore.QRect(10, 10, 441, 421))
        self.deleteTable.setObjectName("deleteTable")
        self.deleteTable.setColumnCount(0)
        self.deleteTable.setRowCount(0)
        self.widget = QtWidgets.QWidget(parent=Widget)
        self.widget.setGeometry(QtCore.QRect(140, 440, 181, 33))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancelBtn = QtWidgets.QPushButton(parent=self.widget)
        self.cancelBtn.setObjectName("cancelBtn")
        self.horizontalLayout.addWidget(self.cancelBtn)
        self.okBtn = QtWidgets.QPushButton(parent=self.widget)
        self.okBtn.setObjectName("okBtn")
        self.horizontalLayout.addWidget(self.okBtn)

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Delete face"))
        self.cancelBtn.setText(_translate("Widget", "Cancel"))
        self.okBtn.setText(_translate("Widget", "Ok"))
