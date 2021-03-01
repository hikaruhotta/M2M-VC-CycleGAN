python -W ignore::UserWarning -m asr.main \
    --name combined_finetune \
    --data_dir ~/data/datasets \
    --save_dir ~/data/results \
    --coraal \
    --voc \
    --num_epochs 50 \
    --batch_size 10 \
    --gpu_ids 2 \
    --num_workers 1 \
