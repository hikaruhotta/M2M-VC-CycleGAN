python -m cycleGAN_VC3.train \
    --name coraal_finetune \
    --save_dir /home/results/cycleGAN_VC3 \
    --num_epochs 30 \
    --normalized_dataset_A_path /home/data/vc3_melspec_dataset/voc_normalized.pickle \
    --norm_stats_A_path /home/data/vc3_melspec_dataset/norm_stat_voc.npz \
    --normalized_dataset_B_path /home/data/vc3_melspec_dataset/coraal_normalized.pickle \
    --norm_stats_B_path /home/data/vc3_melspec_dataset/norm_stat_coraal.npz \
