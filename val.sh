
# trainRun1ValRun3
# python val.py --data /home/bowen68/projects/prairie_dog_2/data/combined_data/run1_run3/run1_run3_yolo.yaml \
# --weights /home/bowen68/projects/prairie_dog_2/yolov5/runs/train/fly1_2/weights/best.pt --img 512 --task val \
# --name trainRun1_last_ValRun3_1  --verbose

# # trainRun3Valrun2
# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/yolov5/weights/512data512model.pt --img 512 --task val \
# --name run1_run2 --verbose --device 7 --save-txt


# # trainRun3ValRun3 (80% 20%)
# python val.py --data /home/bowen68/projects/prairie_dog_2/data/new_data_2024/bisque_data/yolo_anno/run3_yolo.yaml \
# --weights /home/bowen68/projects/prairie_dog_2/yolov5/runs/train/run3/weights/best.pt --img 512 --task val \
# --name trainRun1ValRun3_20percent --verbose

# # trainRun1ValRun3_20percent
# python val.py --data /home/bowen68/projects/prairie_dog_2/data/new_data_2024/bisque_data/yolo_anno/run3_yolo.yaml \
# --weights /home/bowen68/projects/prairie_dog_2/yolov5/runs/train/fly1_2/weights/best.pt --img 512 --task val \
# --name trainRun1ValRun3_20percent --verbose

# test on run3

# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/combined_data/run1_run3/run1_run3_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/yolov5/runs/train/run2_1/weights/best.pt --img 512 --task val \
# --name trainRun3_x_crop_cos_Valrun2 --verbose


# export CUDA_VISIBLE_DEVICES=0 \
# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/yolov5_a100/weights/run12_3con_crop5_200e.pt --img 512 --task val \
# --name cvpr_run12_3con_crop5_200e0_2 --verbose --save-txt --conf-thres 0.2
# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/yolov5_a100/weights/run1run2.pt --img 512 --task val \
# --name cvpr_run1run2_val_0_2 --verbose --save-txt --conf-thres 0.2
# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/yolov5_a100/weights/run12_3con_crop5_200e.pt --img 512 --task test \
# --name cvpr_run12_3con_crop5_200e_test_0_2 --verbose --save-txt --conf-thres 0.2
CUDA_VISIBLE_DEVICES=5
python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
--weights /data/home/bowen/projects/prairie_dog_2/yolov5_a100/runs/train/abb_L13_only/weights/best.pt --img 512 --task test \
--name p3_only --verbose --save-txt

# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/prairieDog2025/runs/train/run12_COScon_aug/weights/best.pt --img 512 --task test \
# --name run12_COScon_aug_conf0.3 --verbose --save-txt --conf-thres 0.3



# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/prairieDog2025/runs/train/run12_KLcon_aug/weights/best.pt --img 512 --task test \
# --name run12_KLcon_aug_conf0.3 --verbose --save-txt --conf-thres 0.3

# python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --weights /data/home/bowen/projects/prairie_dog_2/prairieDog2025/runs/train/run12_MSEcon_aug/weights/best.pt --img 512 --task test \
# --name run12_MSEcon_aug_conf0.3 --verbose --save-txt --conf-thres 0.3
CUDA_VISIBLE_DEVICES=5
python val.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
--weights /data/home/bowen/projects/prairie_dog_2/yolov5_a100/runs/train/abb_L13_only/weights/best.pt --img 512 --task test \
--name p3_only --verbose --save-txt
