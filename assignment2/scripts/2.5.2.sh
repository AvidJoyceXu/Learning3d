echo "2.5 hyperparam tuning: w_chamfer & w_smooth for mesh"
CUDA_VISIBLE_DEVICES=1 python assignment2/train_model.py --type 'mesh' --max_iter 1500 --save_freq 100 --n_point 2000 \
        --w_chamfer 0.8 \
        --w_smooth 0.1

CUDA_VISIBLE_DEVICES=1 python assignment2/eval_model.py --type 'mesh' --load_checkpoint --vis_freq 1000000 --n_point 2000 \
        --w_chamfer 0.8 \
        --w_smooth 0.1