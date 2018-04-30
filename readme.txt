Train with YOLO
python train.py -c config_boat_2.json
	

predict with YOLO
python predict.py -c config_boat_2.json -w full_yolo_retrain_boat.h5
 -i /home/gcx/matlab_scripts/datasets/videos_to_test/lanchaArgos_clip3.avi
