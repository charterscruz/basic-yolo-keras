#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import  TinyYoloTimeDist
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    # width = 1024
    # height = 768
    width = 1920
    height = 1080

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model
    ###############################

    # yolo = YOLO(backend             = config['model']['backend'],
    #             input_size          = config['model']['input_size'],
    #             labels              = config['model']['labels'],
    #             max_box_per_image   = config['model']['max_box_per_image'],
    #             anchors             = config['model']['anchors'])

    yolo = TinyYoloTimeDist(backend=config['model']['backend'],
                            input_size=config['model']['input_size'],
                            labels=config['model']['labels'],
                            max_box_per_image=config['model']['max_box_per_image'],
                            anchors=config['model']['anchors'],
                            input_time_horizon=config['model']['input_time_horizon'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4' or image_path[-4:] == '.avi':
        video_out = image_path[:-4] + '_detected_ty_convlstm' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        # nb_frames = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        #frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = height
        frame_w = width

        video_writer = cv2.VideoWriter(video_out,
                                       cv2.VideoWriter_fourcc(*'MPEG'),
                                       50.0,
                                       (frame_w, frame_h))
        # OPEN RESULTS  file
        results_file = open(image_path[:-4] + '.results_ty_convlstm.txt', 'w+')

        image_seq = np.zeros((config['model']['input_time_horizon'], config['model']['input_size'],
                              config['model']['input_size'], 3))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            image_seq[-1, :, :, :] = cv2.resize(image / 255.,
                                                (config['model']['input_size'],
                                                 config['model']['input_size']))

            if i >= config['model']['input_time_horizon']:

                boxes = yolo.predict(image_seq)
                image = draw_boxes(image, boxes, config['model']['labels'])

                for bb in range(0,len(boxes)):
                    left_coor = boxes[bb].xmin * width
                    right_coor = boxes[bb].xmax * width
                    top_coor = boxes[bb].ymin * height
                    bottom_coor = boxes[bb].ymax * height
                    scoring = boxes[bb].score

                    results_file.write(str(i) + ' ' + str(left_coor) + ' ' + str(top_coor) + ' ' +
                                    str(right_coor - left_coor) + ' ' + str(bottom_coor - top_coor) + ' 1 ' +
                                    str(scoring) + '\n')

            video_writer.write(np.uint8(image))
            image_seq = np.roll(image_seq, (-1, 0, 0, 0))

        results_file.close()
        video_reader.release()
        video_writer.release()

    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
