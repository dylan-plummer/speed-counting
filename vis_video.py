from itertools import product

import math
import numpy as np
import cv2
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import model_from_json, Model
from keras import backend as K
from keras.utils import np_utils

data_dir = os.getcwd() + '/data/'
video_dir = 'speed_videos/'
annotation_dir = 'speed_annotations/'
kernel_size = 8
kernel_frames = 8
frame_size = 32
window_size = 32

vis_frames = 32
vis_size = 32
vis_iter = 200


def video_to_flow_field(video):
    flow = np.array([])
    for i in range(len(video) - 1):
        field = get_flow_field(video, i, i + 1)
        flow = np.append(flow, field)
    return np.reshape(flow, (video.shape[0] - 1, video.shape[1], video.shape[2], 2))


def open_video(file, window_size):
    print('\nOpening', file)
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = frame_size
    frameHeight = frame_size

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = 1

    while True:
        try:
            ret, buf[fc] = cv2.resize(cap.read(), (frame_size, frame_size))
            fc += 1
        except:
            # print('Done reading video')
            break
    print('Done reading video')
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


def plot_weights(weights):
    print(weights.shape)
    for frame_i in range(kernel_frames):
        fig, axes = plt.subplots(8, 4)
        print('Saving frame', frame_i)
        for i in range(len(weights)):
            r = i // 4
            c = i % 4
            img = weights[i][frame_i]
            bgr = cv2.cvtColor(np.uint8(img), cv2.COLOR_HSV2RGB_FULL)
            axes[r][c].imshow(bgr)
            #axes[r][c].quiver(x, y, u, v, mag, edgecolor='k', width=0.05, pivot='tip')
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
        plt.tight_layout(0.5)
        plt.savefig('saved/' + str(frame_i) + '.png', dpi=250)
        plt.clf()


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def plot_conv_layer():
    # load data
    dir = 'models/'
    with open(dir + 'model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(dir + 'model_weights.h5')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    print(layer_dict)

    noise_batch = np.random.random((1, vis_frames, vis_size, vis_size, 3)) * 20.0 + 128.
    filter_index = 31

    layer_name = 'conv3d_2'
    layer_output = layer_dict[layer_name].output
    input_img = model.input
    print(layer_output)
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    loss = K.mean(layer_output[:, :, :, :, filter_index])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    step = 1.
    # run gradient ascent for 20 steps
    for i in range(vis_iter):
        loss_value, grads_value = iterate([noise_batch])
        print(i, 'Loss:', loss_value)
        noise_batch += grads_value * step

    frame_i = 0
    for frame in noise_batch[0]:
        img = deprocess_image(frame)
        img = np.reshape(img, (vis_size, vis_size, 3))
        plt.imshow(img)
        plt.title(layer_name + ': ' + str(filter_index))
        plt.savefig('conv_vis/' + str(frame_i) + '.png')
        plt.clf()
        frame_i += 1


def predict_test():
    dir = 'models/'
    with open(dir + 'model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    # Load weights into the new model
    model.load_weights(dir + 'model_weights.h5')
    video_path = np.random.choice(os.listdir(data_dir + video_dir))
    label_path = data_dir + annotation_dir + video_path.replace('.mp4', '.npy')
    label = np.load(label_path)
    video = open_video(data_dir + video_dir + video_path, window_size=window_size)
    num_clips = 0
    for start_frame in range(0, len(video), window_size):
        if start_frame + window_size < len(video):
            clip = video[start_frame:start_frame + window_size]
            flow_field = video_to_flow_field(clip)
            label_clip = label[np.where(label < start_frame + window_size)]
            label_clip = label_clip[np.where(label_clip > start_frame)]
            y = np.zeros(window_size)
            for frame in label_clip:
                y[frame - start_frame] = 1
            pred = model.predict(np.array([clip]))
            print(pred)
            print(y)
            plt.plot(pred[0])
            plt.plot(y)
            plt.show()
            print(np.sum(pred[0]))


#predict_test()
plot_conv_layer()

