# A robust Pitch-fusion model for Speech Emotion Recognition in tonal languages

### Contact email: thanh.pv.ds@gmail.com

- This is the repository for the submission of the Pitch-fusion model and ViSEC dataset at ICASSP 2024.

- This repository includes:

    + ser_pitch_model.py: implementation of the Pitch-fusion model
    + train_interpolated_pitch.py: training script for the Pitch-fusion model
    + train_no_joint.py: training script for the baseline Wav2Vec 2.0 model

- The following URL leads to the ViSEC dataset download: https://drive.google.com/file/d/1wAK6XcQBZgusyB8sDxlmuC3GhWbNUqCM/view?usp=sharing

    + The visec.zip file contains a wav folder for the audio samples
    + The data.csv file contains the labels of the data
    + Accent refers to the Vietnamese dialects: miền bắc - Northern, miền trung - Central, miền nam - Southern