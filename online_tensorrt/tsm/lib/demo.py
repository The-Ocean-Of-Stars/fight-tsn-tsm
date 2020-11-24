import cv2
import sys
import argparse
import torch
import numpy as np
import os
import time

from Processor import Processor
# from Visualizer import Visualizer


def cli():
    # desc = 'Run TensorRT yolov5 visualizer'
    desc = 'Run TensorRT mobilev2 visualizer'
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument('-model', help='trt engine file located in ./models', required=False)
    # parser.add_argument('--model', default='mobilev22.trt', help='trt engine file located in ./models', required=False)
    parser.add_argument('--model', default='mobilev2.trt', help='trt engine file located in ./models', required=False)
    # parser.add_argument('-image', help='image file path', required=False)
    # parser.add_argument('--image', default='000001.jpg', help='image file path', required=False)
    parser.add_argument('--image', default='sample_720p.jpg', help='image file path', required=False)
    args = parser.parse_args()
    # model = args.model or 'mobilev22.trt'
    model = args.model or 'mobilev2.trt'
    img = args.image or 'sample_720p.jpg'
    return {'model': model, 'image': img}


WINDOW_NAME = 'Video violence Recognition'


def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    # visualizer = Visualizer()

    catigories = [
        "no_violence",  # 0
        "yes_violence"  # 1
    ]

    print("Open camera...")
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('no1_xvid.avi')
    # cap = cv2.VideoCapture('output.avi')
    # cap = cv2.VideoCapture('fi1_xvid.avi')
    cap = cv2.VideoCapture('yes_and_no.mp4')
    # cap = cv2.VideoCapture('reallife.mp4')

    print(cap)

    # set a lower resolution for speed up
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    i_frame = -1
    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        print(img.shape)
        if i_frame % 2 == 0:
            t1 = time.time()
            idx = processor.detect(img)
            print("cls:", catigories[idx])
            t2 = time.time()
            current_time = t2 - t1
            print("current_time", current_time)

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        # t5 = time.time()
        # print("t5-t4:", t5 - t4)
        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()

    # buffer = (
    #     torch.ones((1, 20, 7, 7)),
    #     torch.ones((1, 20, 7, 7)),
    #     torch.ones((1, 12, 14, 14)),
    #     torch.ones((1, 12, 14, 14)),
    #     torch.ones((1, 8, 14, 14)),
    #     torch.ones((1, 8, 14, 14)),
    #     torch.ones((1, 8, 14, 14)),
    #     torch.ones((1, 4, 28, 28)),
    #     torch.ones((1, 4, 28, 28)),
    #     torch.ones((1, 3, 56, 56))
    # )
    # buffer = (
    #     torch.zeros((1, 3, 56, 56)),
    #     torch.zeros((1, 4, 28, 28)),
    #     torch.zeros((1, 4, 28, 28)),
    #     torch.zeros((1, 8, 14, 14)),
    #     torch.zeros((1, 8, 14, 14)),
    #     torch.zeros((1, 8, 14, 14)),
    #     torch.zeros((1, 12, 14, 14)),
    #     torch.zeros((1, 12, 14, 14)),
    #     torch.zeros((1, 20, 7, 7)),
    #     torch.zeros((1, 20, 7, 7))
    # )

    # fetch input
    # print('image arg', args['image'])
    # root = 'inputs/no4_xvi'
    # for filename in os.listdir(root):
    #     t1 = time.time()
    #     print(filename)
    #     # img = cv2.imread('inputs/{}'.format(args['image']))
    #     img = cv2.imread(os.path.join(root, filename))
    #
    #     # inference
    #     # output = processor.detect(img, *buffer)
    #     idx = processor.detect(img)
    #     print("cls:", catigories[idx])
    #
    #     t2 = time.time()
    #     current_time = t2 - t1
    #     print("current_time", current_time)
    #
    #     img = cv2.resize(img, (640, 480))
    #     img = img[:, ::-1]
    #     height, width, _ = img.shape
    #     label = np.zeros([height // 10, width, 3]).astype('uint8') + 255
    #
    #     cv2.putText(label, 'Prediction: ' + catigories[idx],
    #                 (0, int(height / 16)),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.7, (0, 0, 0), 2)
    #     cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
    #                 (width - 170, int(height / 16)),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.7, (0, 0, 0), 2)
    #
    #     img = np.concatenate((img, label), axis=0)
    #     cv2.imshow(WINDOW_NAME, img)
    # cv2.destroyAllWindows()

    # img = cv2.resize(img, (640, 640))
    # object_grids = processor.extract_object_grids(output)
    # visualizer.draw_object_grid(img, object_grids, 0.1)

    # # object visualization
    # object_grids = processor.extract_object_grids(output)
    # visualizer.draw_object_grid(img, object_grids, 0.1)
    #
    # # class visualization
    # class_grids = processor.extract_class_grids(output)
    # visualizer.draw_class_grid(img, class_grids, 0.01)
    #
    # # bounding box visualization
    # boxes = processor.extract_boxes(output)
    # visualizer.draw_boxes(img, boxes)
    #
    # # final results
    # boxes, confs, classes = processor.post_process(output)
    # visualizer.draw_results(img, boxes, confs, classes)


if __name__ == '__main__':
    main()
