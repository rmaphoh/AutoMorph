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
import cleanup

   
if __name__ == "__main__":

    print("Running EyeQ process")
    M0_EQ.EyeQ_process()

    # Eye Quality
    print("Running Image quality assesment")
    M1_EP.M1_image_quality()
    M1_QA.quality_assessment()

    # M2 stages
#    M2_VS.M2_vessel_seg()
#    M2_AV.M2_artery_vein()
#    M2_DC.M2_disc_cup()

    # M3 stages
#    CDDCB.create_data_disc_centred_B()
#    CDDCC.create_data_disc_centred_C()
#    CDMCB.create_macular_centred_B()
#    CDMCC.create_macular_centred_C() 
#    CDMC.create_dataset_macular_centred()

    # cleanup just for azure-run
#    cleanup.cleanup()
