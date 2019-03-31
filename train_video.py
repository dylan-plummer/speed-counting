import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, GlobalMaxPooling3D, Embedding, Flatten, LSTM, TimeDistributed
from keras.layers import SpatialDropout1D
from keras.optimizers import SGD, Adam

from data_helpers import smooth, fit_sin

data_dir = os.getcwd() + '/data/'
video_dir = 'speed_videos/'
annotation_dir = 'speed_annotations/'
learning_rate = 1e-4
batch_size = 8
num_epochs = 50
num_filters = 32
kernel_size = 32
kernel_frames = 4
frame_size = 256
window_size = 5

use_flow_field = True


def video_to_flow_field(video):
    flow = np.array([])
    for i in range(len(video) - 1):
        field = get_flow_field(video, i, i + 1)
        flow = np.append(flow, field)
    return np.reshape(flow, (video.shape[0] - 1, video.shape[1], video.shape[2], 2))


def open_video(file, window_size, flow_field=False):
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = frame_size
    frameHeight = frame_size

    buf = np.zeros((frameCount, frameHeight, frameWidth, 3))

    fc = 0
    ret = 1

    while True:
        try:
            ret, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            buf[fc] = img.copy()
            fc += 1
        except Exception as e:
            break
    cap.release()
    return buf


def get_flow_field(video, i, j):
    prev = cv2.cvtColor(video[i], cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(video[j], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(video[i])
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow


def get_max_length():
    global data_dir
    max_frames = 0
    for filename in os.listdir(data_dir + video_dir):
        label_path = data_dir + annotation_dir + filename.replace('.mp4', '.npy')
        label = np.load(label_path)
        print(label[-1])
        if label[-1] > max_frames:
            max_frames = label[-1]
    return max_frames


def generate_batch(batch_size):
    global data_dir
    x_batch = np.array([])
    y_batch = np.array([])
    total_batch = np.array([])
    while True:
        for filename in os.listdir(data_dir + video_dir):
            video_path = data_dir + video_dir + filename
            label_path = data_dir + annotation_dir + filename.replace('.mp4', '.npy')
            label = np.load(label_path)
            video = open_video(video_path, window_size=window_size)
            num_clips = 0
            random_offset = np.random.randint(0, window_size)
            for start_frame in range(random_offset, len(video), window_size):
                if start_frame + window_size < len(video):
                    clip = video[start_frame:start_frame + window_size]
                    flow_field = video_to_flow_field(np.uint8(clip))
                    label_clip = label[np.where(label < start_frame + window_size)]
                    label_clip = label_clip[np.where(label_clip > start_frame)]
                    y = np.zeros(window_size)
                    for frame in label_clip:
                        y[frame - start_frame] = 1
                    if y.any() == 1:
                        binary_y = 1
                    else:
                        binary_y = 0
                    if use_flow_field:
                        x_batch = np.append(x_batch, flow_field)
                    else:
                        x_batch = np.append(x_batch, clip / 255.)
                    y_batch = np.append(y_batch, binary_y)
                    total_batch = np.append(total_batch, np.sum(y))
                    num_clips += 1
                    if num_clips == batch_size:
                        num_clips = 0
                        if use_flow_field:
                            x_batch = np.reshape(x_batch, (-1, window_size - 1, frame_size, frame_size, 2))
                        else:
                            x_batch = np.reshape(x_batch, (-1, window_size, frame_size, frame_size, 3))
                        #y_batch = np.reshape(y_batch, (-1, 1))
                        yield {'video': x_batch}, {'frames': y_batch}
                        x_batch = np.array([])
                        y_batch = np.array([])
                        total_batch = np.array([])
        print('Trained on all videos.')


if __name__ == '__main__':

    max_frames = get_max_length() + 1
    print('Max Frames:', max_frames)

    if use_flow_field:
        encoder = Input(shape=(window_size - 1, frame_size, frame_size, 2), name='video')
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
    output = Conv3D(128, (1, 3, 3))(output)
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
    output = Dense(1, activation='sigmoid', name='frames')(output)
    model = Model(inputs=encoder,
                  outputs=output)

    adam = Adam(lr=learning_rate)
    sgd = SGD(lr=learning_rate, nesterov=True, decay=1e-6, momentum=0.9)

    losses = {
        'frames': 'mse',
        'count': 'mse',
    }

    loss_weights = {
        'frames': 0.75,
        'count': 0.25,
    }

    model.compile(loss='binary_crossentropy',
                  #loss_weights=loss_weights,
                  optimizer=sgd,
                  metrics=['acc'])

    # model.compile(loss='mse', optimizer='rmsprop')
    print(model.summary())

    history = model.fit_generator(generate_batch(batch_size),
                                  epochs=num_epochs,
                                  steps_per_epoch=100,
                                  verbose=1)

    # Save the weights
    model.save_weights('models/model_weights.h5')

    # Save the model architecture
    with open('models/model_architecture.json', 'w') as f:
        f.write(model.to_json())

    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['total_loss', 'frames_loss', 'count_loss'], loc='upper left')
    plt.show()

    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['frame_acc'], loc='upper left')
    plt.show()

