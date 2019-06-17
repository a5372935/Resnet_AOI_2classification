from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import sys
import glob 
import numpy as np
from Resnet_models.oct_conv2d import *

model_dir = r'C:\Users\ASUS\Desktop\CYCU_CSIE\Resnet18_cifar10\Resnet\Resnet_AOI_2classification\History_model\resnet18_SGD(1e-3)_img_size_100_new'
files = r'C:\Users\ASUS\Desktop\aoiaidata\test_img'

# 載入訓練好的模型
model = load_model(model_dir + '\\GOOD_and_NG_resnet18.h5'
                    #, custom_objects = {'OctConv2D': OctConv2D}
                    )

cls_list = ['Good', 'NG']
acc_all = []
all_target = 0          #全部樣本
pasitive = 0            #全部正樣本

# 辨識每一張圖
for i in range(len(cls_list)):
    acc_target = 0      #單位accuracy
    target = 0          #單位正樣本
    All = 0             #單位數量
    files_path = glob.glob(files + '\\' + str(cls_list[i]) + '\\*.jpg')
    for f in files_path:
        img = image.load_img(f, target_size=(128, 128))
        # if img is None:
        #     continue
        All += 1
        all_target += 1
        # img = cv2.imread(f).astype(np.float32) / 255
        # img = cv2.resize(img, (128, 128))
        img = np.asarray(img, 'f') / 255
        img = np.expand_dims(img, axis = 0)
        pred = model.predict(img)[0]
        pred = pred * 100
        if np.argmax(pred) == i:
            if pred[i] > 95:    
                target += 1
                pasitive += 1
        print(f)
        print(pred)
    acc_target = target / All
    acc_all.append(acc_target)
print(acc_all)              #單位accuracy的集合
print(pasitive/all_target)  #整體accuracy
