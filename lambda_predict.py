from curses import BUTTON1_CLICKED
from io import BytesIO
import boto3
import logging
from glob import glob
from urllib.parse import unquote_plus

import M0_Preprocess.EyeQ_process_main as M0_EQ
import M1_Retinal_Image_quality_EyePACS.test_outside as M1_EP
import M1_Retinal_Image_quality_EyePACS.merge_quality_assessment as M1_QA
import M2_Vessel_seg.test_outside_integrated as M2_VS
import M2_Artery_vein.test_outside as M2_AV
import M2_lwnet_disc_cup.generate_av_results as M2_DC
import M3_feature_whole_pic.retipy.create_datasets_macular_centred as CDMC
import M3_feature_zone.retipy.create_datasets_disc_centred_B as CDDCB
import M3_feature_zone.retipy.create_datasets_disc_centred_C as CDDCC
import M3_feature_zone.retipy.create_datasets_macular_centred_B as CDMCB
import M3_feature_zone.retipy.create_datasets_macular_centred_C as CDMCC

# Define logger class
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event,context):
    s3 = boto3.resource('s3', region_name='us-west-2')

    record = event['Records']
    bucket_name = record['s3']['object']['name']
    bucket = s3.Bucket(bucket_name)
    key = unquote_plus(record['s3']['object']['key']) 

    # temporarily download the images for processing
    tmpkey = key.replace('/', '')
    download_path = '/tmp/images/{}'.format(tmpkey)
    bucket.download_file( key, download_path)
    logger.info(".png download from s3")

#    # Pre-processing
    logger.info('Pre-processing')
    M0_EQ.EyeQ_process()

#    # Eye Quality
    M1_EP.M1_image_quality()
    M1_QA.quality_assessment()

#    #M2 stages
    M2_VS.M2_vessel_seg()
    M2_AV.M2_artery_vein()
    M2_DC.M2_disc_cup()

#    cd ../M3_feature_zone/retipy/
#    python create_datasets_disc_centred_B.py
    CDDCB.create_data_disc_centred_B()

#    python create_datasets_disc_centred_C.py
    CDDCC.create_data_disc_centred_C()

#    python create_datasets_macular_centred_B.py
    CDMCB.create_macular_centred_B()

#    python create_datasets_macular_centred_C.py
    CDMCC.create_macular_centred_C() 

#    python create_datasets_macular_centred.py
    CDMC.create_dataset_macular_centred()

    png = glob("/tmp/results/M0/*.png")[0]
    bucket.upload_file(png, 'M0/test.png')


if __name__ == "__main__":
    lambda_handler()