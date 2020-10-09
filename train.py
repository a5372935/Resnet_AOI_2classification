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

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph',  # log 目錄
                                                #histogram_freq=0,  # 按照何等頻率（epoch）來計算長條圖，0為不計算
                                                batch_size = BATCH_SIZE,     # 用多大量的資料計算長條圖
                                                write_graph=True,   # 是否存儲網路結構圖
                                                write_grads=True,   # 是否視覺化梯度長條圖
                                                write_images=True,  # 是否視覺化參數
                                                )

    ReduceLRCallBack = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',  # monitor：被監測的量
                                                            factor = 0.1,          # factor：每次減少學習率的因數，學習率將以lr = lr*factor的形式被減少
                                                            patience = 5,          # patience：當patience個epoch過去而模型性能不提升時，學習率減少的動作會被觸發
                                                            verbose = 1,           # verbose：資訊展示模式
                                                            mode = 'min',          # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果檢測值觸發學習率減少。在max模式下，當檢測值不再上升則觸發學習率減少
                                                            #epsilon = 0.0001,     # epsilon：閾值，用來確定是否進入檢測值的“平原區”
                                                            cooldown = 0,          # cooldown：學習率減少後，會經過cooldown個epoch才重新進行正常操作
                                                            min_lr = 0.000001      # min_lr：學習率的下限
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
