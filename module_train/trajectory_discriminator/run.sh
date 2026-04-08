srun -p ebench_t --gres=gpu:8  \
torchrun --nproc_per_node=8 video_binary_ddp.py \
    --mode train \
    --videos_true /mnt/inspurfs/ebench_t/acone_data/dis_videos/dis_v6/videos_true \
    --videos_false /mnt/inspurfs/ebench_t/acone_data/dis_videos/dis_v6/videos_false_0214 \
    --eval_videos_true /mnt/inspurfs/ebench_t/acone_data/dis_videos/dis_v6/videos_true_test \
    --eval_videos_false /mnt/inspurfs/ebench_t/acone_data/dis_videos/dis_v6/videos_false_0214_test \
    --sample_rate 20 \
    --max_steps 5000 \
    --eval_every 100 \
    --cache_dir /mnt/inspurfs/ebench_t/acone_data/dis_videos/dis_v6/videos_binary_0224_first \
    --save_dir ckpt_v6_0224_first \
    --num_workers 4 \
    --batch_size 8  \
    --clip_seconds 8 \
    --use_first True