python -W ignore::UserWarning -m asr.main \
    --name combined_vanilla \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --coraal \
    --voc \
    --num_epochs 20 \
    --batch_size 15 \
    --gpu_ids 2,3 \
    --num_workers 1 \
