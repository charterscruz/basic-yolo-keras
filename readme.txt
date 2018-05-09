##Train with YOLO
python train.py -c config_boat_2.json
	

##predict with FULL YOLO
python predict.py
-c config_boat_2.json
-w full_yolo_retrain_boat.h5
 -i /home/gcx/matlab_scripts/datasets/videos_to_test/lanchaArgos_clip3.avi

##predict with tiny YOLO
python predict.py
-c config_boat_tiny_yolo.json
-w tiny_yolo_retrain_boat.h5
-i /home/gcx/matlab_scripts/datasets/videos_to_test/lanchaArgos_clip3.avi

##predict with tiny YOLO +convLSTM
python predict_conv_tnyolo.py
-c configs/config_boat_tiny_yolo_seq.json
-w tiny_yolo_conv_lstm.h5
-i /home/gcx/matlab_scripts/datasets/videos_to_test/2015-04-22-16-05-15_jai_eo.avi

##get features prediction with YOLO
python save_extracted_features.py
-c config_boat_laptop_feature_extraction.json


###TENSORBOARD
##tiny yolo with only one frame at the time
tensorboard --logdir=/home/gcx/logs/tiny_simple --port 6006
#loss after 31 steps = 4.47e-3
#val_loss after the same ammount of steps = 3.67e-3


##full yolo with only one frame at the time
tensorboard --logdir=/home/gcx/logs/full_yolo --port 6006
#loss after 31 steps = 3.182e-3
#val_loss after the same ammount of steps = 2.3968e-3





