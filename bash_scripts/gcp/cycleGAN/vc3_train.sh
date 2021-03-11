python -W ignore::UserWarning -m cycleGAN_VC3.train \
    --name cyclegan_vc3_vcc2018_VCC2SM3_to_VCC2TF1_no_TFAN_1 \
    --save_dir /home/results/cycleGAN_VC3 \
    --num_epochs 6172 \
    --normalized_dataset_A_path /home/sofianzalouk/vcc_2018_melspec/dataset_A_normalized.pickle \
    --norm_stats_A_path /home/sofianzalouk/vcc_2018_melspec/norm_stat_A.npz \
    --normalized_dataset_B_path /home/sofianzalouk/vcc_2018_melspec/dataset_B_normalized.pickle \
    --norm_stats_B_path /home/sofianzalouk/vcc_2018_melspec/norm_stat_B.npz \
    --epochs_per_save 50 \
    --epochs_per_plot 10 \
    --batch_size 1 \
    # --num_frames_validation 320