python -W ignore::UserWarning -m asr.main \
    --name coraal_vanilla \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --coraal \
    --num_epochs 20 \
    --batch_size 15 \
    --gpu_ids 0,1 \
    --num_workers 1 \