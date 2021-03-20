# M2M-VC-CycleGAN
CS224n/224s Class Project

There is a significant performance gap in ASR systems between black and white speakers, which is attributed to insufficient audio data from black speakers available for models to train on. We aim to close this gap by using a CycleGAN based voice converter to generate African American Vernacular English utterances from generic American English utterances as a data augmentation strategy. By using a two-step adversarial loss and a self-supervised frame filling task, we were able to noticeably improve the qualitative performance of our CycleGAN based voice conversion pipeline. In spite of this, we could not establish the method of CycleGAN based voice conversion as a reliable method for data augmentation. While this project was challenging, it was especially rewarding to conduct this line of research which has the ultimate goal of ensuring that marginalized voices are heard.

Train the MaskedCycleGAN-VC model:
```
bash_scripts/aws2/vc3_convert_voc_10.sh
```

Train the ASR model:
```
bash_scripts/aws2/asr_coraal_converted.sh
```
