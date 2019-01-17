import cv2
import os
import numpy as np

data_dir = os.getcwd() + '/unlabeled_videos/'
annotation_dir = os.getcwd() + '/data/speed_annotations/'


def open_video(file, label):
    print('\nOpening', file)
    cap = cv2.VideoCapture(file)

    frame_i = 0
    count = []

    while (cap.isOpened()):
        try:
            ret, frame = cap.read()

            cv2.imshow('frame', frame)
            key = cv2.waitKey(100)
            if key & 0xFF == ord('s'):
                print(frame_i)
                count.append(frame_i)
            elif key & 0xFF == ord('q'):
                print('quitting')
                break
            frame_i += 1
        except Exception as e:
            print('Done reading video')
            break

    cap.release()
    cv2.destroyAllWindows()
    np.save(annotation_dir + label, count)
    return count


def label_videos():
    for filename in os.listdir(data_dir):
        label = filename[:-4]
        print(open_video(data_dir + filename, label))


label_videos()
