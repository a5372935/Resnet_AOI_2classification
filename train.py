import tensorflow as tf
from Resnet_models.res50 import *
from Resnet_models.keras_resnet34 import *
import pickle, os, time

#from ReadMyOwnData import *
# from keras.preprocessing.image import ImageDataGenerator

# def lr_scheduler(epoch):
#     x = 0.2
#     # if epoch >= 75: x /= 5.0
#     # if epoch >= 125: x /= 5.0
#     # if epoch >= 175: x /= 5.0
#     
#     return x

# load_train_dataset = 'GOOD_and_NG_train_v2.tfrecords'
# load_test_dataset = 'GOOD_and_NG_valid_v2.tfrecords'

# def train_tfrecord():
#     # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#     # train_gen = ImageDataGenerator(horizontal_flip=True, 
#     #                                     width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
#     # test_gen = ImageDataGenerator()
#     # y_train = to_categorical(y_train)
#     # y_test = to_categorical(y_test)

#     train_X, train_Y = create_dataset(filepath = load_train_dataset, BATCH_SIZE = 6, NUM_CLASSES = 2)
#     test_x, test_y = create_dataset(filepath = load_test_dataset, BATCH_SIZE = 6, NUM_CLASSES = 2)


#     tf.logging.set_verbosity(tf.logging.FATAL)
#     model = ResNet18(input_tensor= train_X, classes = 2) 
#     model.compile(optimizer = SGD(0.2, momentum=0.9), 
#                     loss = "categorical_crossentropy", 
#                     target_tensors=[train_Y],
#                     metrics = ["acc"]
#                     )

#     model.summary()
#     scheduler = LearningRateScheduler(lr_scheduler)
#     hist = History()
#     start_time = time.time()

#     tbCallBack = TensorBoard(log_dir='./Graph',  # log 目录
#                 #histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                 batch_size=16,     # 用多大量的数据计算直方图
#                 write_graph=True,  # 是否存储网络结构图
#                 write_grads=True,  # 是否可视化梯度直方图
#                 write_images=True, # 是否可视化参数
#                 )
#     model.fit(steps_per_epoch = 300,
#                 shuffle=True,
#                 callbacks=[scheduler, hist, tbCallBack], 
#                 validation_data=(test_x, test_y),
#                 validation_steps = 300,
#                 epochs=200)
                        

#     elapsed = time.time() - start_time
#     print(elapsed)
#     history = hist.history
#     history["elapsed"] = elapsed

#     with open(f"resnet34_AOI.pkl", "wb") as fp:
#         pickle.dump(history, fp)

def train_direct():
    
    DATASET_PATH  = 'C:\\Users\\ASUS\\Desktop\\aoiaidata\\sample'
    IMAGE_SIZE = (128, 128)
    NUM_CLASSES = 2
    # 若 GPU 記憶體不足，可調降 batch size 
    BATCH_SIZE = 16
    # Epoch 數
    NUM_EPOCHS = 125
    # 模型輸出儲存的檔案
    WEIGHTS_FINAL = 'GOOD_and_NG_resnet34.h5'

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                    rescale=1.0/255,
                                                                    width_shift_range=0.2,
                                                                    height_shift_range=0.2,
                                                                    shear_range=0.2,
                                                                    zoom_range=0.2,
                                                                    channel_shift_range=10,
                                                                    horizontal_flip=True,
                                                                    fill_mode='nearest')
    train_batches = train_datagen.flow_from_directory(DATASET_PATH + '\\train',
                                                    target_size=IMAGE_SIZE,
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE,)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
    valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '\\valid',
                                                    target_size=IMAGE_SIZE,
                                                    shuffle=False,
                                                    batch_size=BATCH_SIZE)

    for cls, idx in train_batches.class_indices.items():
        print('Class #{} = {}'.format(idx, cls))

    tf.logging.set_verbosity(tf.logging.FATAL) #訊息紀錄
    
    model = ResNet34(input_shape = (IMAGE_SIZE[0],IMAGE_SIZE[1],3), classes = NUM_CLASSES)
    
    model.compile(optimizer = tf.keras.optimizers.SGD(lr =1e-3), 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
                )
    model.summary()
    #scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    hist = tf.keras.callbacks.History()
    start_time = time.time()

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph',  # log 目录
                                                #histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                                batch_size = BATCH_SIZE,     # 用多大量的数据计算直方图
                                                write_graph=True,  # 是否存储网络结构图
                                                write_grads=True,  # 是否可视化梯度直方图
                                                write_images=True, # 是否可视化参数
                                                )

    ReduceLRCallBack = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',  # monitor：被监测的量
                                                            factor = 0.1,         # factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                                            patience = 5,          # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                                            verbose = 1,           # verbose：信息展示模式
                                                            mode = 'auto',         # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
                                                            #epsilon = 0.0001,     # epsilon：阈值，用来确定是否进入检测值的“平原区”
                                                            cooldown = 0,          # cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                                            min_lr = 0.0000001         # min_lr：学习率的下限
                                                            )
    model.fit_generator(generator = train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        callbacks=[hist, tbCallBack], 
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs=NUM_EPOCHS)
                        
    elapsed = time.time() - start_time
    print(elapsed)
    history = hist.history
    history["elapsed"] = elapsed
    model.save(WEIGHTS_FINAL)

    with open(f"resnet34_AOI.pkl", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    train_direct()