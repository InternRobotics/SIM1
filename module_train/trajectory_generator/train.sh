srun -p ebench_t --gres=gpu:8  \
torchrun --nproc_per_node=8 train.py \
    --mode train \
    --data_root ./dataset \
    --out_dir ./output \
    --batch_size 64 \
    --epochs 200000 \
    --lr 1e-4 \
    --save_every 1000 \
    --log_every 50 \