python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name debug \
    --save_dir /home/results/cycleGAN_VC3 \
    --num_epochs 50 \
    --normalized_dataset_A_path /home/data/vc3_melspec_dataset/voc_normalized.pickle \
    --norm_stats_A_path /home/data/vc3_melspec_dataset/norm_stat_voc.npz \
    --normalized_dataset_B_path /home/data/vc3_melspec_dataset/coraal_normalized.pickle \
    --norm_stats_B_path /home/data/vc3_melspec_dataset/norm_stat_coraal.npz \
    --epochs_per_save 10 \
    --epochs_per_plot 1 \

    # --continue_train
