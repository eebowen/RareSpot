# python -m torch.distributed.run --nproc_per_node 2 --master_port=29502 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --epochs 200 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 128 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run1_2_200_nwd --device 4,5

# python -m torch.distributed.run --nproc_per_node 8 \
# train.py --data /home/bowen68/projects/prairie_dog_2/data/new_data_2024/yolo/run3_yolo.yaml \
# --epochs 100 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 128 \
# --hyp data/hyps/hyp.scratch-high.yaml --name run3_no_black --device 0,1,2,3,4,5,6,7

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=29501 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/new_data_2024/yolo/run3_yolo_w_crop.yaml \
# --epochs 100 --imgsz 512 --cfg yolov5x.yaml --weights models/yolov5x.pt --batch-size 128 \
# --hyp data/hyps/hyp.scratch-high.yaml --name run3_x_crop --device 2,3


# python -m torch.distributed.run --nproc_per_node 2 --master_port=29502 \
# train.py - -data /data/home/bowen/projects/prairie_dog_2/data/new_data_2024/yolo/run3_yolo.yaml \
# --epochs 300 --imgsz 512 --weights models/yolov5l.pt --cfg yolov5l.yaml  --batch-size 128 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run3_l_cos --device 2,3
### NWD

# python -m torch.distributed.run --nproc_per_node 4 --master_port=29502 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_1024_a100.yaml \
# --epochs 200 --imgsz 1024 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 64 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run1_2_1024 --device 4,5,6,7

# python -m torch.distributed.run --nproc_per_node 2 --master_port=29503 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --epochs 200 --imgsz 1024 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 32 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run1_2_1024model_512data --device 2,3

# python -m torch.distributed.run --nproc_per_node 2 --master_port=29501 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run2_run1_yolo_a100.yaml \
# --epochs 200 --imgsz 1024 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 32 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run2_1_1024model_512data --device 2,3


# python -m torch.distributed.run --nproc_per_node 4 --master_port=29502 \
# train.py --data /data/home/bowen/data/prairie_dog/run1run2/run1_run2_yolo_a100.yaml \
# --epochs 130 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 60 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run12_w_consistency --device 1,3,4,5

# run12_w_consistency

# python train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --epochs 200 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 1 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name debug --device 2

# python -m torch.distributed.run --nproc_per_node 4 --master_port=29502 \
# train.py --data /data/home/bowen/data/prairie_dog/run1run2/run2_run1_yolo_a100.yaml \
# --epochs 130 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 60 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name run21_con --device 1,3,4,5



# python -m torch.distributed.run --nproc_per_node 4 --master_port=29502 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/mara/yolo_512/mara.yaml \
# --epochs 100 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml --batch-size 128  \
# --hyp data/hyps/hyp.scratch-high.yaml --name mara_3con --device 3,5,6,7

# python train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --epochs 200 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l_p3_only.yaml  --batch-size 60 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name abb_p3_only --device 4


# export WANDB_ENTITY=eebowenz
# python -m torch.distributed.run --nproc_per_node 4 --master_port=29501 \
# train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
# --epochs 200 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l_p3_only.yaml  --batch-size 60 --cos-lr \
# --hyp data/hyps/hyp.scratch-high.yaml --name abb_p3_only --device 3,5,6,7


export WANDB_ENTITY=eebowenz
python -m torch.distributed.run --nproc_per_node 2 --master_port=29502 \
train.py --data /data/home/bowen/projects/prairie_dog_2/data/old_data/two_classes/run1_run2_yolo/run1_run2_yolo_a100.yaml \
--epochs 200 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l_p3_only.yaml  --batch-size 60 --cos-lr \
--hyp data/hyps/hyp.scratch-high.yaml --name abb_L13_only --device 1,2
