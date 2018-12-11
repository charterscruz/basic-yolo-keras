#! /usr/bin/env python

import argparse
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-d',
    '--display',
    type=bool,
    help='display images or not')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    display      = args.display

    width = 1920
    height = 1080

    # width = 1024
    # height = 768

    if display:
        cv2.namedWindow('img', 0)
        cv2.resizeWindow('img', 192, 108)

    results_string = '_3_3_box3'

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4' or image_path[-4:] == '.avi':
        video_out = image_path[:-4] + '_' + results_string + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        frame_h = height
        frame_w = width

        if int(cv2.__version__[0]) == 3:
            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            video_writer = cv2.VideoWriter(video_out,
                                           cv2.VideoWriter_fourcc(*'MPEG'),
                                           50.0,
                                           (frame_w, frame_h))
        else:
            nb_frames = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            # video_writer = cv2.VideoWriter(video_out,
            #                                cv2.VideoWriter_fourcc(*'MPEG'),
            #                                50.0,
            #                                (frame_w, frame_h))
            # video_writer = cv2.VideoWriter(video_out, fourcc=None, fps=50, frameSize=(frame_w, frame_h))

        # OPEN RESULTS  file
        results_file = open(image_path[:-4] + results_string + '.results.txt', 'w+')

        # full_tensor = np.empty((config['model']['time_horizon'] * config['model']['time_stride'],
        #                         config['model']['input_size'],
        #                         config['model']['input_size'],
        #                         3))

        full_tensor = np.empty((config['model']['input_size'],
                                config['model']['input_size'],
                                3))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            full_tensor[:, :, :] = cv2.resize(image,
                                                  (config['model']['input_size'], config['model']['input_size']))

            # if i >= config['model']['time_horizon'] * config['model']['time_stride']:

                # time_sampled_tensor = full_tensor[::-config['model']['time_stride'], :, :, :]

            boxes = yolo.predict(full_tensor[:, :, :])
            image = draw_boxes(image, boxes, config['model']['labels'])

            if display:
                cv2.imshow('img', image)
                cv2.waitKey(1)

            for bb in range(0,len(boxes)):
                left_coor = boxes[bb].xmin * width
                right_coor = boxes[bb].xmax * width
                top_coor = boxes[bb].ymin * height
                bottom_coor = boxes[bb].ymax * height
                scoring = boxes[bb].score

                results_file.write(str(i) + ' ' + str(left_coor) + ' ' + str(top_coor) + ' ' +
                                   str(right_coor - left_coor) + ' ' + str(bottom_coor - top_coor) + ' 1 ' +
                                   str(scoring) + '\n')

            # full_tensor = np.roll(full_tensor, -1, axis=0)

            if int(cv2.__version__[0]) == 3:
                video_writer.write(np.uint8(image))
            else:
                pass

        results_file.close()
        if int(cv2.__version__[0]) == 3:
            video_writer.release()
        else:
            pass

        video_reader.release()

    else:
        print('could not find video. Exiting!')
        sys.error('could not find video. Exiting!')
        # image = cv2.imread(image_path)
        # boxes = yolo.predict(image)
        # image = draw_boxes(image, boxes, config['model']['labels'])

        # print(len(boxes), 'boxes are found')
        #
        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
