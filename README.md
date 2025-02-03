how to uhhh run it ig 
1. install python \
   1a. wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tar.xz \
   1b. tar -xf Python-3.9.5.tar.xz \
   1c. cd Python-3.9.5 \
   1d. ./configure \
   1e. make -j$(nproc) \
   1f. sudo make install 
2. clone this repository \
   2a. git clone https://github.com/choccymalk/midas-with-yolov5.git \
   2b. cd midas-with-yolov5 
3. make a virtualenvironment \
   3a. python3.9 -m venv venv \
   3b. source venv/bin/activate 
4. install requirements.txt \
   4a. pip install -r requirements.txt 
5. print this image i think it will work if you take a picture of your screen with it but you should probably just print it 
   ![print this](https://github.com/choccymalk/midas-with-yolov5/blob/main/chessboard.jpg?raw=true) 
6. lay the chessboard flat and run take-pictures-for-calibration.py make sure you get enough with a lot of different angles like you need to take about 30 ish 
7. calibrate your camera with depth-with-object-detection-midas.py \
   7a. python depth-with-object-detection-midas.py -c 1 
8. now you can run it!!! \
   8a. python depth-with-object-detection-midas.py -c 0 
