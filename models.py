from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, AveragePooling3D, Dropout
from keras.layers import BatchNormalization, Input, GlobalMaxPooling3D, Embedding, Flatten, LSTM, TimeDistributed
from keras.layers import concatenate, GlobalAveragePooling3D
from keras.initializers import glorot_uniform, Constant

kernel_init = glorot_uniform()
bias_init = Constant(value=0.2)


def stacked_model(use_flow_field, grayscale, window_size, frame_size):
    if use_flow_field:
        encoder = Input(shape=(window_size - 1, frame_size, frame_size, 2), name='video')
    elif grayscale:
        encoder = Input(shape=(window_size, frame_size, frame_size, 1), name='video')
    else:
        encoder = Input(shape=(window_size, frame_size, frame_size, 3), name='video')
    output = Conv3D(4, (2, 16, 16))(encoder)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling3D((1, 2, 2))(output)
    output = Conv3D(8, (2, 8, 8))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling3D((1, 2, 2))(output)
    output = Conv3D(16, (1, 4, 4))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling3D((1, 2, 2))(output)
    output = Conv3D(32, (1, 3, 3))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling3D((1, 2, 2))(output)
    output = Conv3D(64, (1, 3, 3))(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = MaxPooling3D((1, 2, 2))(output)

    #output = TimeDistributed(MaxPooling2D(pool_size=(4, 4)))(output)
    #output = Conv3D(num_filters, (kernel_frames, kernel_size, kernel_size), activation='relu')(encoder)
    #output = MaxPooling3D(pool_size=(2, 2, 2), strides=(4, 2, 2))(output)
    #output = Conv3D(64, (3, 3, 3), activation='relu')(output)
    #output = MaxPooling3D(pool_size=(2, 2, 2), strides=(3, 2, 2))(output)

    #output = LSTM(100, input_shape=(batch_size, 1, window_size - 1, 32, 32, 2))(output)
    #output = Conv3D(128, (2, 2, 2), activation='relu')(output)
    #output = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(output)
    #output = TimeDistributed(Dense(256, activation='relu'))(output)
    #output = TimeDistributed(Dense(512, activation='relu'))(output)
    #output = TimeDistributed(Dense(128, activation='relu'))(output)
    output = Flatten()(output)
    output = Dense(256, activation='relu')(output)
    #output = LSTM(50)(output)
    #output = TimeDistributed(Flatten())(output)
    #repetitions = Dense(1, activation='sigmoid', name='count')(output)
    output = Dense(window_size, activation='softmax', name='frames')(output)
    model = Model(inputs=encoder,
                  outputs=output)
    return model


def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    conv_1x1 = Conv3D(filters_1x1, (2, 1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv3D(filters_3x3_reduce, (2, 1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv3D(filters_3x3, (2, 3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv3D(filters_5x5_reduce, (1, 1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv3D(filters_5x5, (2, 5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPooling3D((1, 3, 3), strides=(1, 1, 1), padding='same')(x)
    pool_proj = Conv3D(filters_pool_proj, (2, 1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=4, name=name)

    return output


def build_inception_model(use_flow_field, window_size, frame_size):
    if use_flow_field:
        input_layer = Input(shape=(window_size - 1, frame_size, frame_size, 2), name='video')
    else:
        input_layer = Input(shape=(window_size, frame_size, frame_size, 3), name='video')

    x = Conv3D(16, (2, 7, 7), padding='same', strides=(2, 2, 2), activation='relu', name='conv_1_7x7/2',
               kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPooling3D((1, 3, 3), padding='same', strides=(2, 2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv3D(16, (2, 1, 1), padding='same', strides=(1, 1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv3D(64, (1, 3, 3), padding='same', strides=(1, 1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPooling3D((1, 3, 3), padding='same', strides=(2, 2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=16,
                         filters_3x3_reduce=32,
                         filters_3x3=64,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=32,
                         filters_3x3_reduce=64,
                         filters_3x3=64,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = MaxPooling3D((1, 3, 3), padding='same', strides=(2, 2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=128,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=64,
                         name='inception_4a')

    x1 = AveragePooling3D((1, 5, 5), strides=3)(x)
    x1 = Conv3D(128, (1, 1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(1, activation='sigmoid', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=128,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=64,
                         filters_3x3=128,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')

    x = inception_module(x,
                         filters_1x1=32,
                         filters_3x3_reduce=64,
                         filters_3x3=64,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_4d')

    x2 = AveragePooling3D((1, 5, 5), strides=3)(x)
    x2 = Conv3D(128, (1, 1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(1, activation='sigmoid', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                         filters_1x1=32,
                         filters_3x3_reduce=32,
                         filters_3x3=64,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_4e')

    x = MaxPooling3D((1, 3, 3), padding='same', strides=(2, 2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=16,
                         filters_3x3_reduce=64,
                         filters_3x3=32,
                         filters_5x5_reduce=64,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=64,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    x = GlobalAveragePooling3D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')
    print(model.summary())

    return model