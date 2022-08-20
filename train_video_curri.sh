CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/9 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/8 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/7 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/6 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/5 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/4 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/3 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/2 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/1 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts
CUDA_VISIBLE_DEVICES=0  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/0 \
    --expname train_video_beef_curri3  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri3/ckpts

