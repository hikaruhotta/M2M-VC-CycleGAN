python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name cyclegan_vc3_src_28_tgt_DCB_se2_ag3_m_02_1 \
    --save_dir /home/results/cycleGAN_VC3 \
    --num_epochs 1000 \
    --normalized_dataset_A_path /home/data/vc3_melspec_dataset/voc_normalized.pickle \
    --norm_stats_A_path /home/data/vc3_melspec_dataset/norm_stat_voc.npz \
    --normalized_dataset_B_path /home/data/vc3_melspec_dataset/coraal_normalized.pickle \
    --norm_stats_B_path /home/data/vc3_melspec_dataset/norm_stat_coraal.npz \
    --epochs_per_save 50 \
    --epochs_per_plot 25 \
    --batch_size 20
    # --continue_train
