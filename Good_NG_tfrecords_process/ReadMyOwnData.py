import tensorflow as tf
#入参example_proto也就是tf_serialized

# def parse_tf(example_proto):
#         #定义解析的字典
#         # dics = {}
#         # dics['label'] = tf.FixedLenFeature(shape=[],dtype = tf.int64)
#         # dics['img_raw'] = tf.FixedLenFeature(shape=[],dtype = tf.string)
        
#         #调用接口解析一行样本
#         parsed_example = tf.parse_single_example(serialized = example_proto,
#             features =
#                 {
#                     "label": tf.FixedLenFeature([], tf.int64), 
#                     "img_raw": tf.FixedLenFeature([], tf.string)
#                 }
#         )
#         image = tf.decode_raw(parsed_example['img_raw'], tf.uint8)
#         #image = tf.reshape(image, [128, 128])
        
#         #这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
#         # 有了这里的归一化处理，精度与原始数据一致
#         image = tf.cast(image, tf.float64) * (1.0 / 255)
#         label = parsed_example['label']
#         label = tf.cast(label, tf.float32)
#         #label = tf.one_hot(label, depth = 2, on_value=0)
#         return image, label


# def read_dataset(filenames, batchSize):
#     dataset = tf.data.TFRecordDataset(filenames)
    
#     dataset = dataset.map(parse_tf)
#     dataset = dataset.shuffle(10)
#     dataset = dataset.repeat()
#     dataset = dataset.batch(batchSize)
#     iterator = dataset.make_one_shot_iterator()
#     X, Y = iterator.get_next()

#     X = tf.reshape(X, [-1, 128, 128, 3])
#     Y = tf.reshape(Y, [-1, 1])

#     return X, Y




def create_dataset(filepath, BATCH_SIZE, NUM_CLASSES):
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
        features = {'img_raw': tf.FixedLenFeature([], tf.string),
                    "label": tf.FixedLenFeature([], tf.int64)}
        # Load one example
        parsed_features = tf.parse_single_example(proto, features)
        # Turn your saved image string into an array
        parsed_features['img_raw'] = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
        parsed_features['img_raw'] = tf.reshape(parsed_features['img_raw'] , [128, 128, 3])
        parsed_features['img_raw'] = tf.cast(parsed_features['img_raw'], tf.float32) * (1. / 255)
        parsed_features['label'] = tf.one_hot(parsed_features['label'], NUM_CLASSES)
        # parsed_features['label'] = tf.reshape(parsed_features['label'], [1])
        # parsed_features['label'] = tf.cast(parsed_features['label'], tf.float32)
        #parsed_features['img_raw'] = tf.image.convert_image_dtype(parsed_features['img_raw'],tf.float32)

        return parsed_features['img_raw'], parsed_features["label"]


    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)
    # This dataset will go on forever
    dataset = dataset.repeat()
    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(5)
    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    #image = tf.reshape(image, [-1, 128, 128, 3])
    # Create a one hot array for your labels
    #label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# load_train_dataset = 'GOOD_and_NG_train_v2.tfrecords'
# train_X, train_Y = create_dataset(filepath = load_train_dataset, BATCH_SIZE = 6, NUM_CLASSES = 2)
# print(train_Y)