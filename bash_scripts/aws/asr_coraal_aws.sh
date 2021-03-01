python -W ignore::UserWarning -m asr.main \
    --name coraal_finetune \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --coraal \
    --num_epochs 50 \
    --batch_size 10 \
    --gpu_ids 1 \
    --num_workers 1 \
