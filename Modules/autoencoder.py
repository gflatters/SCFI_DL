from keras import backend
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers import Dense, BatchNormalization, Flatten, Dropout, Reshape, MaxPooling3D, ConvLSTM2D, MaxPooling2D,LSTM,TimeDistributed, UpSampling2D
from keras.optimizers import SGD,Adadelta
from keras.backend import squeeze

def autoencoder_2d(input_len, input_wid):
    input_shape=(input_len,input_wid,1)
    num_classes = 2
    #encoder
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
     
    #decoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    decoded = model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    model.compile(
            loss=binary_crossentropy,
            optimizer=Adadelta(),
            metrics=['accuracy'],
        )
    model.summary()
    return decoded
