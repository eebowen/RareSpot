
CUDA_VISIBLE_DEVICES=5
python val.py --data PATH_TO_run1_run2_yolo.yaml \
--weights PATH_TO_WEIGHTS --img 512 --task test  \
--name val --verbose --save-txt
