from curses import BUTTON1_CLICKED
from io import BytesIO
import boto3
import logging
from glob import glob
from urllib.parse import unquote_plus
import json
import os

import automorph.M0_Preprocess.EyeQ_process_main as M0_EQ
import automorph.M1_Retinal_Image_quality_EyePACS.test_outside as M1_EP
import automorph.M1_Retinal_Image_quality_EyePACS.merge_quality_assessment as M1_QA
import automorph.M2_Vessel_seg.test_outside_integrated as M2_VS
import automorph.M2_Artery_vein.test_outside as M2_AV
import automorph.M2_lwnet_disc_cup.generate_av_results as M2_DC
import automorph.M3_feature_whole_pic.retipy.create_datasets_macular_centred as CDMC
import automorph.M3_feature_zone.retipy.create_datasets_disc_centred_B as CDDCB
import automorph.M3_feature_zone.retipy.create_datasets_disc_centred_C as CDDCC
import automorph.M3_feature_zone.retipy.create_datasets_macular_centred_B as CDMCB
import automorph.M3_feature_zone.retipy.create_datasets_macular_centred_C as CDMCC
import automorph.config as gv
import imghdr

# Define logger class
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# get s3 client
s3 = boto3.client('s3', region_name='us-west-2')

def lambda_handler(event,context):

    record = event['Records'][0] # records [0]
    bucket = record['s3']['bucket']['name']
    key = unquote_plus(record['s3']['object']['key']) 
    tmpkey = key.split('/')[-1]
    download_path = gv.image_dir
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    s3.download_file(bucket, key, download_path+tmpkey)
    logger.info("object download from s3")

    # quick png check
    ftype = imghdr.what(download_path+tmpkey)
    if ftype != 'png':
        logger.info("{} is not a png file, aborting".format(key))
        return {'statusCode': 200, 'headers':{'Content-type':'application/json'},
        'body': "uploaded object {} was not a png filetype".format(key)
    }

    # Pre-processing
    logger.info('Pre-processing')
    M0_EQ.EyeQ_process()

    # Eye Quality
    M1_EP.M1_image_quality()
    M1_QA.quality_assessment()

#    #M2 stages
#    logger.info("starting_vessel_seg")
#    M2_VS.M2_vessel_seg()
#    logger.info("starting AV seg")
#    M2_AV.M2_artery_vein()
#    logger.info("starting disc cup seg")
#    M2_DC.M2_disc_cup()
#
#    CDDCB.create_data_disc_centred_B()
#
#    CDDCC.create_data_disc_centred_C()
#
#    CDMCB.create_macular_centred_B()
#
#    CDMCC.create_macular_centred_C() 
#
#    CDMC.create_dataset_macular_centred()
    logger.info("finished pipeline")

    logger.info("copying files")

    output_bucket = "eyeact-automorph2"
    sub_dir = tmpkey.split('.png')[0]

    for root, dir, files in os.walk(gv.results_dir):
       for f in files:
            png = os.path.join(root, f)
            id = os.path.join(sub_dir, root.split(gv.results_dir)[-1], f) # need to change for lee lab install
            s3.upload_file(png, output_bucket, id)

    return {
        'statusCode': 200,
        'headers':{
            'Content-type':'application/json'
        },
        'body': "finished"
    }

#with open("/data/anand/AutoMorph_Lee/test_event.json", 'r') as f:
    #data = json.load(f)
#lambda_handler(data, 1)
