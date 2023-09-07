# face-recognition-app
* ### Works on Linux, Windows, MacOS


<div align="center">
<h2>How does the program look like?</h2>
<a href="https://ibb.co/XWPcYM6"><img src="https://i.ibb.co/JcTD3g8/Screenshot-2023-09-07-at-20-52-30.png" alt="MainWindow" border="0"></a><br>
<a href="https://ibb.co/t3NJfXR"><img src="https://i.ibb.co/6NpWhBQ/Screenshot-2023-09-07-at-20-52-47.png" alt="AddWindow" border="0"></a><br>
<a href="https://ibb.co/yNmgxQw"><img src="https://i.ibb.co/tHwY1J7/Screenshot-2023-09-07-at-20-52-53.png" alt="DeleteWindow" border="0"></a><br>
</div>


## Installation

### Install on Windows
* First you need to install [Visual Studio Community 2019](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads)
* Then install [MySQL Community Server](https://dev.mysql.com/downloads/mysql)

### Install on MacOS
* Install [MySQL Community Server](https://dev.mysql.com/downloads/mysql)

### Install on Linux

* Install MySQL Community Server
```
sudo apt update
sudo apt -y install mysql-server
sudo systemctl start mysql.service
```

* Configure MySQL Community Server
```
sudo mysql
```
```
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'abcd1234';
exit
```

### Finally install the libraries
```
pip install cmake && pip install face-recognition mysql-connector-python==8.0.33 opencv-python "PyQt6-sip<13.5" "PyQt6-Qt6<6.5" "PyQt6<6.5"
```
