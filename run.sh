#!/bin/bash  
# RUN FILE FOR AUTOMORPH
# YUKUN ZHOU 2023-08-24
# Updated: 2025-06-21

date

# ----------------------------- #
# Parse optional arguments
# ----------------------------- #
NO_PROCESS=0
NO_QUALITY=0
NO_SEGMENTATION=0
NO_FEATURE=0

for arg in "$@"; do
  case $arg in
    --no_process)
      NO_PROCESS=1
      shift
      ;;
    --no_quality)
      NO_QUALITY=1
      shift
      ;;
    --no_segmentation)
      NO_SEGMENTATION=1
      shift
      ;;
    --no_feature)
      NO_FEATURE=1
      shift
      ;;
  esac
done

# ----------------------------- #
# Step 0 - Prepare AUTOMORPH_DATA directory and clean up results
# ----------------------------- #

python automorph_data.py

if [ -z "${AUTOMORPH_DATA}" ]; then
  rm -rf ./Results/*
  echo "AUTOMORPH_DATA not set, using default directory"
else
  rm -rf "${AUTOMORPH_DATA}/Results"/*
  echo "AUTOMORPH_DATA set to ${AUTOMORPH_DATA}"
fi

# ----------------------------- #
# Step 1 - Image Preprocessing
# ----------------------------- #
if [ $NO_PROCESS -eq 0 ]; then
  echo "### Preprocess Start ###"
  cd M0_Preprocess
  python EyeQ_process_main.py
  cd ..
else
  echo "### Skipping Preprocessing ###"
fi

# ----------------------------- #
# Step 2 - Image Quality Assessment
# ----------------------------- #
if [ $NO_QUALITY -eq 0 ]; then
  echo "### Image Quality Assessment ###"
  cd M1_Retinal_Image_quality_EyePACS
  sh test_outside.sh
  python merge_quality_assessment.py
  cd ..
else
  echo "### Skipping Image Quality Assessment ###"
fi

# ----------------------------- #
# Step 3 - Segmentation Modules
# ----------------------------- #
if [ $NO_SEGMENTATION -eq 0 ]; then
  echo "### Segmentation Modules ###"
  
  cd M2_Vessel_seg
  sh test_outside.sh
  cd ..

  cd M2_Artery_vein
  sh test_outside.sh
  cd ..

  cd M2_lwnet_disc_cup
  sh test_outside.sh
  cd ..
else
  echo "### Skipping Segmentation Modules ###"
fi

# ----------------------------- #
# Step 4 - Feature Measurement
# ----------------------------- #
if [ $NO_FEATURE -eq 0 ]; then
  echo "### Feature Measuring ###"

  cd M3_feature_zone/retipy/
  python create_datasets_disc_centred_B.py
  python create_datasets_disc_centred_C.py
  python create_datasets_macular_centred_B.py
  python create_datasets_macular_centred_C.py
  cd ../..

  cd M3_feature_whole_pic/retipy/
  python create_datasets_macular_centred.py
  python create_datasets_disc_centred.py
  cd ../..

  python csv_merge.py
else
  echo "### Skipping Feature Measurement ###"
fi

echo "### Done ###"
date
