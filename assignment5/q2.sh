# CUDA_VISIBLE_DEVICES=1 python train.py --task seg

python eval_seg.py --load_checkpoint best_model --output_dir ./output/seg