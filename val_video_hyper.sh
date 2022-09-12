CUDA_VISIBLE_DEVICES=3  python val_owndata.py  \
    --dataset_name MyDataset --datadir  ./video_data/beef/11  \
    --expname val_hyper_beef_3 --with_rgb_loss  --batch_size 256  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --net_type v3  --netdepth 7 \
    --ckpt ./runs_fine_tuning/train_hyper_beef_3/ckpts/latest.tar --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_hyper_beef_3/ckpts \
    
