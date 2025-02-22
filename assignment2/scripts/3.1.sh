echo "3.1 Training a convolutional occupancy network"

CUDA_VISIBLE_DEVICES=0 python assignment2/train_model.py --type 'occupancy' --max_iter 2000 --save_freq 200

echo "2.1 evaluation"

CUDA_VISIBLE_DEVICES=0 python assignment2/eval_model.py --type 'occupancy'  --load_checkpoint --vis_freq 1000000 --n_point 1000