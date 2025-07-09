python -m torch.distributed.run --nproc_per_node 2 --master_port=29501 \
train.py --data PATH_TO_run1_run2_yolo.yaml \
--epochs 200 --imgsz 512 --weights 'models/yolov5l.pt' --cfg yolov5l.yaml  --batch-size 32 --cos-lr \
--hyp data/hyps/hyp.scratch-high.yaml --name train --device 2,3