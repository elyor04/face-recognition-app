# face-recognition-app
* ### Works on Linux, Windows, MacOS


## Installation

### Install on Windows
* First you need to install [Visual Studio Community 2019](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads)
* Then install [MySQL Community Server](https://dev.mysql.com/downloads/mysql)

### Install on MacOS
* Install [MySQL Community Server](https://dev.mysql.com/downloads/mysql)

### Install on Linux

#### Install MySQL Community Server
```
sudo apt update
sudo apt -y install mysql-server
sudo systemctl start mysql.service
```

#### Configure MySQL Community Server
```
sudo mysql
```
```
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'abcd1234';
```

### Finally install the libraries
```
pip install cmake && pip install face-recognition mysql-connector-python opencv-python PyQt6
```
