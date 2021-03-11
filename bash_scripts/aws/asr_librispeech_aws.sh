python -W ignore::UserWarning -m asr.main \
    --name librispeech \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --librispeech \
    --num_epochs 100 \
    --batch_size 20 \
    --gpu_ids 2 \
    --num_workers 1 \
    --n_feats 80 \
