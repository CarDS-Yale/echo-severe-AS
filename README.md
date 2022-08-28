# Automated detection of severe aortic stenosis using single-view echocardiography: A self-supervised ensemble learning approach

Code repository for "Automated detection of severe aortic stenosis using single-view echocardiography: A self-supervised ensemble learning approach" by Gregory Holste, Evangelos K. Oikonomou, Bobak Mortazavi, Kamil F. Faridi, Edward J. Miller, Robert L. McNamara, Harlan M. Krumholz, Zhangyang Wang, and Rohan Khera.

-----

## Abstract

<p align=center>
    <img src=figs/echo_avs_fig1_v6.png height=600>
</p>

Early diagnosis of aortic stenosis (AS) is critical for the timely deployment of invasive therapies. We hypothesized that self-supervised learning of parasternal long axis (PLAX) videos from transthoracic echocardiography (TTE) could extract discriminative features to identify severe AS without Doppler imaging. In a training set of 5,311 studies (17,601 videos) from 2016-2020, we performed self-supervised pretraining based on contrastive learning of PLAX videos, then used those learned weights to initialize a convolutional neural network to predict severe AS in an external set of 2,040 studies from 2021. Our model achieved an AUC of 0.97 (95% CI: 0.96-0.99) for detecting severe AS with 95.8% sensitivity and 90% specificity. The models were interpretable with saliency maps identifying the aortic valve as the predictive region. Among non-severe AS cases, predicted probabilities were associated with worse quantitative metrics of AS suggesting association with AS severity. We propose an automated approach for screening for severe AS using single-view 2D echocardiography, with implications for point-of-care screening. 

-----

## Pipeline

```
# Create conda environments
conda env create -f view_cls.yml
conda env create -f echo.yml

conda activate echo

## PREPROCESSING ##
# Load videos from DICOM format, deidentify frames, and save to AVI format
cd preprocessing/
python deidentify.py --data_dir <path_to_dicom_data> --output_dir <path_to_output_avis> --n_jobs <num_cpus>

# Run videos through view classifier to identify PLAX clips
conda deactivate
conda activate view_cls
python classify_view.py --data_dir <path_to_output_avis> --output_dir <path_to_output_plax_prob_csv> --csv_name <plax_prob_csv_name>

# More thoroughly mask peripheral pixels, downsample frames, and randomly split internal data into train/val/test
conda deactivate
conda activate echo
python preprocess.py --output_dir <path_to_preprocessed_data> --label_csv_path <path_to_label_csv> --plax_csv_path <path_to_plax_prob_csv>

## SELF-SUPERVISED LEARNING (SSL) ##
# Perform SSL pretraining on training set. Model trained for 300 epochs on two NVIDIA RTX 3090 GPUs.
cd ../ssl_pretraining/
python main.py --data_dir <path_to_preprocessed_data> --out_dir <path_to_results> --model_name <model_name>

## SUPERVISED FINETUNING ON AS DETECTION TASK ##
cd ../AS_detection/

# Finetune model from SSL initialization
python main.py \
    --output_dir <path_to_ssl_model_results> \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --ssl <path_to_ssl_model_directory> \
    --lr 0.1

# Finetune model from Kinetics-400 initialization
python main.py \
    --output_dir <path_to_kinetics_model_results> \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment

# Finetune model from random initialization
python main.py \
    --output_dir <path_to_random_model_results> \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --rand_init

## ANALYSIS AND VISUALIZATION ##
# Run GradCAM on representative videos (5 true positives, 1 true negative, 1 false positive) and save output
python gradcam.py --data_dir <path_to_preprocessed_data> --out_dir <path_to_ssl_gradcam_output> --model_name ssl
python gradcam.py --data_dir <path_to_preprocessed_data> --out_dir <path_to_kinetics_gradcam_output> --model_name kinetics
python gradcam.py --data_dir <path_to_preprocessed_data> --out_dir <path_to_random_gradcam_output> --model_name random

# From GradCAM output, apply 2D max projection and overlay on frame from each video
python viz_gradcam.py --data_dir <path_to_preprocessed_data> --model_name ssl --gradcam_dir <path_to_ssl_gradcam_output>
python viz_gradcam.py --data_dir <path_to_preprocessed_data> --model_name kinetics --gradcam_dir <path_to_kinetics_gradcam_output>
python viz_gradcam.py --data_dir <path_to_preprocessed_data> --model_name random --gradcam_dir <path_to_random_gradcam_output>

# From GradCAM outputs for each model, generate frame-by-frame saliency maps. Save video showing overlaid saliency maps side by side by side.
python viz_gradcam_videos.py --data_dir <path_to_preprocessed_data> --gradcam_dirs <path_to_ssl_gradcam_output> <path_to_kinetics_gradcam_output> <path_to_random_gradcam_output>

# Get main results on internal and external test sets presented in Table 1.
python get_main_results.py --split test
python get_main_results.py --split ext_test

# Get video-level results on internal and external test sets presented in Extended Data Table 2.
python get_main_results.py --split test --video_level
python get_main_results.py --split ext_test --video_level

# Get internal test set results by number of PLAX videos per study used to form a single "study-level" AS prediction
python get_results_by_num_videos.py --split test

# Plot ROC and precision-recall (PR) curves
python plot_curves.py
```
