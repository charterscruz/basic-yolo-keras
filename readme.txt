Train with YOLO
python train.py -c config_boat_2.json
	

predict with FULL YOLO
python predict.py -c config_boat_2.json -w full_yolo_retrain_boat.h5
 -i /home/gcx/matlab_scripts/datasets/videos_to_test/lanchaArgos_clip3.avi

predict with tiny YOLO
python predict.py -c config_boat_tiny_yolo.json
-w tiny_yolo_retrain_boat.h5
-i /home/gcx/matlab_scripts/datasets/videos_to_test/lanchaArgos_clip3.avi

get features prediction with YOLO
python save_extracted_features.py -c config_boat_laptop_feature_extraction.json
