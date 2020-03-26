## Crop Your Face
A weird game that lets you make your face fly away when you do a "pistol gesture".

### 1. What's it like?
Like this?\
![image](./imgs/demo.gif)

### 2. Package Requirements
* tensorflow==2.1.0 (I don't know if TF1 could work. Probably will?)
* dlib==19.18.0
* numpy
* cv2
* csv

Use `requirements.txt` to auto install by typing this at your terminal:
```
$ pip install -r requirements.txt
```

### 3. Model Requirements
Most of the models are included in the `model/` folder. However, you still need to download a large one from dlib.\
If your computer supports curl command, try running `setup.sh` although I don't know if it works :)
```
$ bash ./setup.sh
```
If it doesn't work, you can manually download the model from 
[here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and put it inside `model/`\
(Don't rename it!)
  
### 4. Get started
Run the main file:
```
$ python main.py
```
And wait for your webcam to launch.

### 5. Details

#### Face detection
Two models are used, both from [dlib](http://dlib.net).\
`get_frontal_face_detector()` is a detector that gives us the bounding box of faces.\
`shape_predictor`, applied on dlib's pretrained 68 landmarks model gives us the landmarks of detected faces.\
Landmarks are indexed in the following order.\
![image](./imgs/face%20landmark.png)

#### Palm detection
Two models are used, both from [mediapipe](https://github.com/google/mediapipe/).\
`palm_detection.tflite` gives us the bounding box of detected palm.\
`hand_landmark.tflite` gives us the landmarks of detected faces.\
Landmarks are indexed in the following order.\
![image](./imgs/hand%20landmark.jpg)

### 6. Others
In fact, `palm_detection.tflite` from mediapipe cannot be loaded alone 
because there's some custom operators defined in their repo, 
and I don't want to dig into it too much.

However, thanks to [metalwhale's awesome repo](https://github.com/metalwhale/hand_tracking),
who provided us a modified version of the model: `palm_detection_without_custom_op.tflite`
so that combined with his code from `hand_tracker.py`, I was able to get the model to work.

