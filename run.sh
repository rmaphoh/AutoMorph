#!/bin/bash  
# RUN FILE FOR AUTOMORPH
# YUKUN ZHOU 2023-08-24

date
# STEP 0 - prepare AUTOMORH_DATA directory and clean up results
if [ -z "${AUTOMORPH_DATA}" ]; then
  export AUTOMORPH_DATA="$(dirname "$(realpath "$0")")"
  echo "AUTOMORPH_DATA not set, using default: ${AUTOMORPH_DATA}"
fi

python automorph_data.py

rm -rf ${AUTOMORPH_DATA}/Results/*

# STEP 1 IMAGE PREPROCESSING (EXTRA BACKGROUND REMOVE, SQUARE)

echo "### Preprocess Start ###"
cd M0_Preprocess
python EyeQ_process_main.py

# STEP 2 IMAGE QUALITY ASSESSMENT

echo "### Image Quality Assessment ###"

cd ../M1_Retinal_Image_quality_EyePACS
sh test_outside.sh

python merge_quality_assessment.py

# STEP 3 OPTIC DISC & VESSEL & ARTERY/VEIN SEG
echo "### Segmentation Modules ###"

cd ../M2_Vessel_seg
sh test_outside.sh

cd ../M2_Artery_vein
sh test_outside.sh

cd ../M2_lwnet_disc_cup
sh test_outside.sh

# STEP 4 METRIC MEASUREMENT
echo "### Feature measuring ###"

cd ../M3_feature_zone/retipy/
python create_datasets_disc_centred_B.py
python create_datasets_disc_centred_C.py
python create_datasets_macular_centred_B.py
python create_datasets_macular_centred_C.py

cd ../../M3_feature_whole_pic/retipy/
python create_datasets_macular_centred.py
python create_datasets_disc_centred.py

cd ../../
python csv_merge.py

echo "### Done ###"


date
