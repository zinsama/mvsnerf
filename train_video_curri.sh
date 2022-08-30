CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/9 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 5e-5\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/8 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 1e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0 python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/7 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 1e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/6 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 1e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/5 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 3e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/4 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 3e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/3 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 3e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/2 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 5e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/1 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 5e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts
CUDA_VISIBLE_DEVICES=0  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/0 \
    --expname train_video_beef_curri7  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 5e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_beef_curri7/ckpts

