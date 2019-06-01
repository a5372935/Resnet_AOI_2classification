from keras import backend as K
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
import sys
import glob 
import numpy as np


files = glob.glob('C:\\Users\\ASUS\\Desktop\\aoiaidata\\test_img\\NG\\*.jpg')

# 載入訓練好的模型
model = load_model('GOOD_and_NG_resnet34.h5')

#cls_list = ['Good', 'NG']
NG = 0
all_NG = 0

# 辨識每一張圖
for f in files:
    img = image.load_img(f, target_size=(128, 128))
    # if img is None:
    #     continue
    all_NG += 1
    # img = cv2.imread(f).astype(np.float32) / 255
    # img = cv2.resize(img, (128, 128))
    img = np.asarray(img, 'f') / 255
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img)[0]
    pred = pred *100
    if pred[1] > pred[0]:
        NG += 1
    print(f, pred)
print(NG / all_NG)