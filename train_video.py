import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization, Input, GlobalMaxPooling3D, Embedding, Flatten, LSTM, TimeDistributed
from keras.layers import SpatialDropout1D
from keras.optimizers import SGD, Adam

from data_helpers import smooth, fit_sin
from models import stacked_model, build_inception_model

data_dir = os.getcwd() + '/data/'
video_dir = 'speed_videos/'
annotation_dir = 'speed_annotations/'
learning_rate = 5e-4
batch_size = 16
num_epochs = 10
num_filters = 32
kernel_size = 32
kernel_frames = 4
frame_size = 64
window_size = 8

use_flow_field = False
grayscale = True


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

    if grayscale:
        buf = np.zeros((frameCount, frameHeight, frameWidth), dtype=np.uint8)
    else:
        buf = np.zeros((frameCount, frameHeight, frameWidth, 3))

    fc = 0
    ret = 1

    while True:
        try:
            ret, img = cap.read()
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
            buf[fc] = img.copy()
            fc += 1
        except Exception as e:
            break
    cap.release()
    if grayscale:
        buf = np.reshape(buf, (frameCount, frameHeight, frameWidth, 1))
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


def get_total_frames():
    global data_dir
    total = 0
    for filename in os.listdir(data_dir + video_dir):
        label_path = data_dir + annotation_dir + filename.replace('.mp4', '.npy')
        label = np.load(label_path)
        print(label[-1])
        total += label[-1]
    return total


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
                    if use_flow_field:
                        flow_field = video_to_flow_field(np.uint8(clip))
                        label_clip = label[np.where(label < start_frame + window_size - 1)]
                        label_clip = label_clip[np.where(label_clip > start_frame)]
                        y = np.zeros(window_size - 1)
                    else:
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
                    y_batch = np.append(y_batch, y)
                    total_batch = np.append(total_batch, np.sum(y))
                    num_clips += 1
                    if num_clips == batch_size:
                        num_clips = 0
                        if use_flow_field:
                            x_batch = np.reshape(x_batch, (-1, window_size - 1, frame_size, frame_size, 2))
                        elif grayscale:
                            x_batch = np.reshape(x_batch, (-1, window_size, frame_size, frame_size, 1))
                        else:
                            x_batch = np.reshape(x_batch, (-1, window_size, frame_size, frame_size, 3))
                        if use_flow_field:
                            y_batch = np.reshape(y_batch, (-1, window_size - 1))
                        else:
                            y_batch = np.reshape(y_batch, (-1, window_size))
                        yield {'video': x_batch}, {'frames': y_batch}
                        x_batch = np.array([])
                        y_batch = np.array([])
                        total_batch = np.array([])
        print('Trained on all videos.')


if __name__ == '__main__':

    total_frames = get_total_frames() + 1
    print('Total Frames:', total_frames)
    print('Total Samples:', total_frames // window_size)

    model = stacked_model(use_flow_field, grayscale, window_size, frame_size)
    #model = build_inception_model(use_flow_field, window_size, frame_size)

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
                  metrics=['accuracy'])

    # model.compile(loss='mse', optimizer='rmsprop')
    print(model.summary())

    history = model.fit_generator(generate_batch(batch_size),
                                  epochs=num_epochs,
                                  steps_per_epoch=100,
                                  verbose=2)

    # Save the weights
    model.save_weights('models/model_weights.h5')

    # Save the model architecture
    with open('models/model_architecture.json', 'w') as f:
        f.write(model.to_json())

    print(history.history.keys())

    # summarize history for accuracy
    for loss_plot in history.history.keys():
        if 'loss' in loss_plot:
            plt.plot(history.history[loss_plot], label=loss_plot)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

    for acc_plot in history.history.keys():
        if 'acc' in acc_plot:
            plt.plot(history.history[acc_plot], label=acc_plot)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

