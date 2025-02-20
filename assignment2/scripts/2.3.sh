echo "2.3 training"
# Increasing n_point for better visualization results
CUDA_VISIBLE_DEVICES=2 python assignment2/train_model.py --type 'mesh' --max_iter 1500 --save_freq 100 --n_point 2000 \
        --w_chamfer 1.0 \
        --w_smooth 0.1

echo "2.3 evaluation"
CUDA_VISIBLE_DEVICES=2 python assignment2/eval_model.py --type 'mesh' --load_checkpoint --vis_freq 1000000 --n_point 2000