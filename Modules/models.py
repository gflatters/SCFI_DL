from keras import backend
from keras import regularizers
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv3D
from keras.layers import Dense, BatchNormalization, Flatten, Dropout, Reshape, MaxPooling3D, ConvLSTM2D, MaxPooling2D,LSTM,TimeDistributed
from keras.optimizers import SGD,Adadelta, Adam, RMSprop
from keras.backend import squeeze


def get_luke_model(input_len):
    input_shape=(input_len,input_len,1)
    num_classes=2
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
    
        # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
            loss=binary_crossentropy,
            optimizer=Adadelta(),
            metrics=['accuracy'],
        )
    return model

def get_luke_model_140(input_shape=(100,140,1)):
    #change input shape and learning rate
    
    num_classes=2
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
    
        # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
            loss=binary_crossentropy,
            optimizer=SGD(),
            metrics=['accuracy'],
        )
    return model

def get_luke_model_140_2(input_shape=(100,140,1)):
    #change input shape and learning rate
    
    num_classes=2
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
        # this converts our 3D feature maps to 1D feature vectors
        
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
            loss=binary_crossentropy,
            optimizer=SGD(),
            metrics=['accuracy'],
        )
    return model

def get_luke_model_noBN(input_len, input_wid):
    input_shape=(input_len,input_wid,1)
    num_classes=2
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer = regularizers.l2(l = 0.0005), padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer = regularizers.l2(l = 0.0005), padding='same'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
        # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(
            loss=binary_crossentropy,
            optimizer=Adadelta(),
            metrics=['accuracy'],
        )
    return model

def get_3d_model(input_size):
    num_classes=2
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(1,1,10), input_shape=input_size, activation='relu', padding='valid'))
    model.add(Reshape((80,80,32)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
        # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
            loss=binary_crossentropy,
            optimizer=Adadelta(),
            metrics=['accuracy'],
        )
    return model


def get_luke_3d_model(num_classes=2, input_shape=(140, 140, 1), n_filters=32, sequence_length=20):
    shape = (sequence_length, input_shape[0], input_shape[1], input_shape[2])

    model = Sequential()
    model.add(Conv3D(n_filters, kernel_size=(3, 3, 3),
                     padding='same',  activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=True, dropout=0.5))
    model.add(LSTM(128, return_sequences=True, dropout=0.2))
    model.add(LSTM(32, dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=binary_crossentropy,
        optimizer=Adadelta(lr=0.1),
        metrics=['accuracy'],
    )
    return model

def get_luke_3d_model2(num_classes=2, input_shape=(140, 140, 1), n_filters=32, sequence_length=20):
    shape = (sequence_length, input_shape[0], input_shape[1], input_shape[2])

    model = Sequential()
    model.add(Conv3D(n_filters, kernel_size=(3, 3, 3),
                     padding='same',  activation='relu', input_shape=shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Conv3D(n_filters, (3, 3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.32))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    #model.add(Dropout(0.4))
    
    model.add(TimeDistributed(Flatten()))
    #model.add(LSTM(256, return_sequences=True, dropout=0.2))
    model.add(LSTM(128, return_sequences=True, dropout=0.2))
    model.add(LSTM(32, dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss=binary_crossentropy,
        optimizer=Adadelta(),
        metrics=['accuracy'],
    )
    return model


def model_lstm2d(num_classes=3, input_shape=(140, 140, 1), n_filters=32, sequence_length=20):
    shape = (sequence_length, input_shape[0], input_shape[1], input_shape[2])

    model = Sequential()

    model.add(ConvLSTM2D(n_filters, (3, 3), activation='relu',
                         padding='same', input_shape=shape, return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))
    model.add(ConvLSTM2D(n_filters, (3, 3), activation='relu',
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Dropout(0.25))
    model.add(ConvLSTM2D(n_filters*2, (3, 3), activation='relu',
                         padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
            loss=binary_crossentropy,
            optimizer=Adadelta(),
            metrics=['accuracy'],
    )

    return model
