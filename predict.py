#! /usr/bin/env python

import argparse
import os
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

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    # width = 1024
    # height = 768
    width = 1920
    height = 1080

    cv2.namedWindow('img', 0)

    major = int(cv2.__version__.split(".")[0])


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
        video_out = image_path[:-4] + '_detected_ty' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        if major == 3:
            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            nb_frames = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        #frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = height
        frame_w = width

        if major == 3:
            video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        else:
            pass
            # video_writer = cv2.VideoWriter(video_out,
            #                                cv2.VideoWriter_fourcc(*'MPEG'),
            #                                50.0,
            #                                (frame_w, frame_h))

        # OPEN RESULTS  file
        results_file = open(
            image_path[:-4] + '.results_ty.txt', 'w+')


        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            # print boxes
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

            # video_writer.write(np.uint8(image))
            cv2.imshow('img', image)
            cv2.waitKey(1)

        results_file.close()
        video_reader.release()
        # video_writer.release()

    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
