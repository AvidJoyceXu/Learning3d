echo "2.1 training"
# Increasing n_point for better visualization results
CUDA_VISIBLE_DEVICES=0 python assignment2/train_model.py --type 'vox' --max_iter 1500 --save_freq 100

echo "2.1 evaluation"
# Doesn't support batch size > 1
CUDA_VISIBLE_DEVICES=0 python assignment2/eval_model.py --type 'vox'  --load_checkpoint --vis_freq 1000000 --n_point 1000