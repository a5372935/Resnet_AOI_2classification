import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model


def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding='same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X
 
def convolutional_block(X, f, filters, stage, block, s = 2):
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X
 
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (f, f), strides = (s,s),padding='same',name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
 
    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1,1),padding='same',name=conv_name_base+'2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F2, (1, 1), strides=(s,s),padding='valid',name=conv_name_base+'1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
 
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    return X
    
# GRADED FUNCTION: ResNet18
 
def ResNet18(image_shape, NUM_CLASSES):
    # Define the input as a tensor with shape input_shape
    X_input = Input(image_shape)
 
    # Zero-Padding
    #X = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(X_input)
    # Stage 1
    #X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X_input)
    X = Conv2D(64, (3, 3), padding = 'same', name = 'conv1')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    #X = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(X)
    #X = MaxPooling2D((3, 3), strides=(2, 2))(X)
 
    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64], stage = 2, block='b')
 
    ### START CODE HERE ###
 
    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128], stage = 3, block='b')
 
    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256,256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256,256], stage = 4, block='b')
 
    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3,filters= [512,512], stage = 5, block = 'a', s = 2)
    X = identity_block(X, 3, [512,512], stage = 5, block='b')
 
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D()(X)
    #X = GlobalAveragePooling2D()(X)

    # output layer
    X = Flatten()(X)
    X = Dense(NUM_CLASSES , activation='softmax')(X)
    
    model = Model(X_input, X)
 
    return model

# model = ResNet18(input_shape=(32,32,3),classes=10)
# model.summary()