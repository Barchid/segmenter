python main.py --batch_size=4 --default_root_dir="experiments/deit_seg_b_mask_16_nyuv2" --data_dir=data/nyuv2 --config_path="pretrain_configs/deit_seg_b_mask_16.yml" --learning_rate=0.001 --max_epochs=70 --dataset="nyudv2" --gpus=1 --auto_select_gpus 

python main.py --default_root_dir="experiments/tiny_mask_16_nyuv2" --data_dir=data/nyuv2 --config_path="pretrain_configs/seg_t_mask_16.yml" --learning_rate=0.001 --max_epochs=70 --dataset="nyudv2" --gpus=1 --auto_select_gpus

python main.py --batch_size=4 --default_root_dir="experiments/seg_s_mask_16_nyuv2" --data_dir=data/nyuv2 --config_path="pretrain_configs/seg_s_mask_16.yml" --learning_rate=0.001 --max_epochs=70 --dataset="nyudv2" --gpus=1 --auto_select_gpus 