echo "2.2 training"
# Increasing n_point for better visualization results
CUDA_VISIBLE_DEVICES=1 python assignment2/train_model.py --type 'point' --max_iter 1500 --save_freq 100 --n_point 2000

echo "2.2 evaluation"
CUDA_VISIBLE_DEVICES=1 python assignment2/eval_model.py --type 'point' --load_checkpoint --vis_freq 100000 --n_point 2000