# AutoMorph 2022 👀
--Code for [AutoMorph: Automated Retinal Vascular Morphology Quantification via a Deep Learning Pipeline](https://tvst.arvojournals.org/article.aspx?articleid=2783477).

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.

Project website: https://rmaphoh.github.io/projects/automorph.html

Talks on NIHR Moorfields BRC: https://moorfieldsbrc.nihr.ac.uk/case-study/research-report/



## News 👀
2025-05-16 update: Docker and its instruction have been updated.

2024-06-27 update: pytorch 2.3 & python 3.11 supported; Mac M2 GPU supported; CPU supported (thanks to [staskh](https://github.com/staskh))

2023-08-24 update: Added feature measurement for disc-centred images; removed unused files.
&nbsp;




## Pixel resolution

The units for vessel average width, disc/cup height and width, and calibre metrics are defined as microns. For it, we need to organise a [resolution_information.csv](https://github.com/rmaphoh/AutoMorph/blob/main/resolution_information.csv) which includes the pixel resolution information, which can be queried in FDA or Dicom files. Alternatively, approximate value can be used, e.g., 0.008 for Topcon 3D-OCT.

**If you don't use these features or care their units**, you can put all images in the folder ./images and run

```bash
python generate_resolution.py
```
&nbsp;


## Running AutoMorph
### Running with Colab

Use the Google Colab and a free Tesla T4 gpu [Colab link click](https://colab.research.google.com/drive/13Qh9umwRM1OMRiNLyILbpq3k9h55FjNZ?usp=sharing).

👀 A specific version for [APTOSxJSAIO 2025 hands-on tutorial](https://colab.research.google.com/drive/1ppzxBElLa_yJl-V3IWKyQQDG_2m6OWVL#scrollTo=vnEudwJimFgt)


### Running on local/virtual machine

Install and use on your own machines [LOCAL.md](LOCAL.md).


### Running with Docker

Zero experience in Docker? No worries [DOCKER.md](DOCKER.md).

&nbsp;

## Common questions

### Environment variables
A few optional environment variables are available for all pipelines:
- AUTOMORPH_DATA : the directory where the Results are stored. If not defined, the "Results" subdirectory is created in the current directory. If AUTOMORPH_DATA defined outside of source directory (for example, /tmp/AutoMorh ) source directory can be made read-only - for deployment inside of AWS Lambda.
- NUM_WORKERS : defines a number of workers for the dataloader. The default is 8. If NUM_WORKERS is set to 0, the dataloader will be single-threaded.

### Memory/ram error

We use Tesla T4 (16Gb) and 32vCPUs (120Gb). When you meet memory/ram issue in running, try to decrease batch size:

* ./M1_Retinal_Image_quality_EyePACS/test_outside.sh -b=64 to smaller, e.g., 32 or 16.
* ./M2_Artery_vein/test_outside.sh --batch-size=8 to smaller
* ./M2_lwnet_disc_cup/test_outside.sh --batchsize=8 to smaller


### Invalid results

In csv files, invalid values (e.g., optic disc segmentation failure) are indicated with NAN values.  


### Components

1. Vessel segmentation [BF-Net](https://github.com/rmaphoh/Learning-AVSegmentation.git)

2. Image pre-processing [EyeQ](https://github.com/HzFu/EyeQ.git) 

3. Optic disc segmentation [lwnet](https://github.com/agaldran/lwnet.git)

4. Feature measurement [retipy](https://github.com/alevalv/retipy.git)

&nbsp;

## Citation

```
@article{zhou2022automorph,
  title={AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline},
  author={Zhou, Yukun and Wagner, Siegfried K and Chia, Mark A and Zhao, An and Xu, Moucheng and Struyven, Robbert and Alexander, Daniel C and Keane, Pearse A and others},
  journal={Translational vision science \& technology},
  volume={11},
  number={7},
  pages={12--12},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```

