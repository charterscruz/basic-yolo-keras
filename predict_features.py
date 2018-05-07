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
import glob
import os

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
    # width = 1920
    # height = 1080
    width = 1920
    height = 1080

    map_size = 13

    time_horizon = 10

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
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

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
        results_file = open(
            image_path[:-4] + '.results.txt', 'w+')


        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
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

        results_file.close()
        video_reader.release()
        video_writer.release()

    else:

        for feature_img in glob.glob(image_path):
            image = np.zeros(time_horizon, map_size, map_size, 1024)
            predict = True
            for tim_idx in range(0, time_horizon):
                # check if image exists
                name_to_load = str(int(feature_img) + tim_idx)
                if os.path.isfile(name_to_load):
                    # load feature "image"
                    image[tim_idx, :, :, :] = np.load()
                else:
                    predict = False
                    break

            if predict:
                #predict based on that image feature sequence
                boxes = yolo.predict(image)
            #load ground truth
            # compare predictions with GT
            image = draw_boxes(image, boxes, config['model']['labels'])

        print(len(boxes), 'boxes are found')

        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
