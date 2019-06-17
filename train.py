import tensorflow as tf
from Resnet_models.res18 import *
from Resnet_models.Oct_res18 import *
import pickle, os, time


def train(Alpha):
    
    DATASET_PATH  = r'C:\Users\ASUS\Desktop\AI_sample'
    IMAGE_SIZE = (128, 128)
    NUM_CLASSES = 2
    # 若 GPU 記憶體不足，可調降 batch size 
    BATCH_SIZE = 100
    # Epoch 數
    NUM_EPOCHS = 200
    # 模型輸出儲存的檔案
    WEIGHTS_FINAL = 'GOOD_and_NG_Oct_resnet18.h5'
    #Best_Weights_Filepath = './best_weights_GOOD_and_NG_resnet50.hdf5'

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
    
    model = Oct_ResNet18(input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), classes = NUM_CLASSES, pooling = 'avg', alpha = Alpha)
    #model = ResNet18(input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), classes = NUM_CLASSES, pooling = 'avg')

    model.compile(optimizer = tf.keras.optimizers.SGD(lr = 1e-3, momentum = 0.9), 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
                )
    model.summary()

    hist = tf.keras.callbacks.History()
    start_time = time.time()

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph',  # log 目录
                                                #histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                                batch_size = BATCH_SIZE,     # 用多大量的数据计算直方图
                                                write_graph=True,   # 是否存储网络结构图
                                                write_grads=True,   # 是否可视化梯度直方图
                                                write_images=True,  # 是否可视化参数
                                                )

    ReduceLRCallBack = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',  # monitor：被监测的量
                                                            factor = 0.1,          # factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
                                                            patience = 5,          # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
                                                            verbose = 1,           # verbose：信息展示模式
                                                            mode = 'min',          # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少
                                                            #epsilon = 0.0001,     # epsilon：阈值，用来确定是否进入检测值的“平原区”
                                                            cooldown = 0,          # cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
                                                            min_lr = 0.000001      # min_lr：学习率的下限
                                                            )
    
    model_EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_acc',
                                                        patience = 15,
                                                        verbose = 1,
                                                        mode = 'max')
    model.fit_generator(generator = train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        callbacks=[hist, tbCallBack, model_EarlyStopping, ReduceLRCallBack],
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs=NUM_EPOCHS)
                        
    elapsed = time.time() - start_time
    print(elapsed)
    history = hist.history
    history["elapsed"] = elapsed
    model.save(WEIGHTS_FINAL)

    with open(f"Oct_resnet18_AOI.pkl", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    train(Alpha = 0.25)
