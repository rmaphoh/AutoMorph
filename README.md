# AutoMorph 2022 👀
--Code for [AutoMorph: Automated Retinal Vascular Morphology Quantification via a Deep Learning Pipeline](https://www.medrxiv.org/content/10.1101/2022.05.26.22274795v1.full.pdf).

![badge-logo](https://img.shields.io/badge/CMIC-2022-orange.svg?style=flat-square&logo=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEARwBHAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9KfH3xK8N/DHRzqfiTVodNtzny0c5lmYfwxoPmY/QfXFfInxW/ax+JfjHw7qur/DTSf8AhHdC0tRONU1GNHkuiHA2YbKYOcFVyfVhXU/tdafbar8cvhRaXlvHc2s3mpLDKuVdfMTII7jjpXS/E7Q9MHwR1qMqjXt3amG0tBhc4kX5Y0HsDXqKNLD0Y1JR5pS77LW2x81Uq4nGYqWHpy5Ywte272e/z6HE/Af/AIKMeHfFdzD4f+JlongbxHkRi+Yn+zp26ZLHmEn/AGsr/t19i29zFdwRzwTLNDKodJI2DK6nkEEcEEd6/F74uaRb22hX+6ENJCqshkX54yWAIB6iv1W/ZjGP2c/hmAMAeHbHgf8AXFa5KkYuKnHQ9XD1anO6VR3t1PM/2wPhX498S6p4X8ZeA7O11W/8PLIX0+Rv3z5ZWDRqcK+MH5cg9MZ6V8t6P8dxc3d3B4gmn0fWbcstza6sGBRh1AyAQR/dIB9jX6gEda84+KP7PPgD4x3VrceKfD8N7eWzKy3kTNDOyg/6tpEIZkPTaSfbFaQrxlFU6qultY562ClGpKvh5Wk977PofnHceGvEX7UWo3Gh+CvDs2ogsqXWu3H7mGEA5+ZzwBx0OWPZa/TX4S+C7n4efC7wn4XurmK6utH0u3sZZ4QQkjRxhSyg84JHGa3PDfhrSvCej22laLp1tpWm267YrW0iEcaD2A/n1NalZVaqmlGKskdOGwzo3nOV5M//2Q==)  ![badge-logo](https://img.shields.io/badge/Moorfields-2022-blue.svg?style=flat-square&logo=data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAAaABsDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9M/G/jC08E6G+o3au43rFHGmN0jt0UbiBk4OASNxwo+YgH528UfGrWJHmW+8Sf2RKYXlGn6TC9zcIogLnKIodW2zeYgk2F1SM4VlcV6B8cHkk8U+Go3jLW6LI6EJI26Q9EGyMFixVAIvOXzM7dpyDXnfwU8M23i69it9S1K8064lsItSmW3nkt7i6uGmcsGZ5HkKwMTGsZciPe6HJPy/R4OjSp0PbVFf+vn27Hz2Lq1atb2NN2/r5fmdT4V+N+o2VwJr3ULXXtFe5lhkurfObci4kjKgEK7YZZEHynf5USIGeRivvtrdxXltHPC4kikUMrLyCK+Pruwh03xs1rDcG9t7qS90uS7ZXY30EFoJIHf8A0crO8Su6IXl2tvZmYsSB7h8LvEF6/gHR/JxdQiNlSWOBipUOwAXYVUAAYAVQAAAAKzx2FgoqpTVr/wDB/wAv+AaYLEyu6dR3t/wP8zX+Lnga58U6da32lDZrWnsXgkjVPNZepRGbHJIXALKhOC4dQVPzRPodra+ZBGbDTdOUmaXRb6zmEVojvHHILV4BG8S/Z55HSIjc0lyHbazfL9rt2+tYHiTQ9Nv4fNutPtbmX5RvmhV2xuBxkj1VT+A9KywOLlT/AHXQ0x2EjP8Ae9T5d8LeEdQ1fVZLe1K6jrc0f2G7udOtza21jEzb5fIUMu6N54pJCZgG8xZFEiNLG5+qvD2gR6HolnYFhcNBGFaaUFmdupYk5JySTkkn1JPNTaJp9rYaZbLbW0NurRqSIowoJ2gdvYAfgK0R0Fc+MxUq8uXZI6MHho0Y827Z/9k=)


Project website: https://rmaphoh.github.io/projects/automorph.html

Talks on NIHR Moorfields BRC: https://www.moorfieldsbrc.nihr.ac.uk/news/automorph-tool-to-analyse-retinal-photographs

Before starting, we summarise the features for three running ways:

* Google Colab (no commands/code, free gpu for 12 hours)
* Configure environment on local/virtual machine (data privacy, code development)
* Docker image (data privacy, no need to configure environment)


## Index

- [AutoMorph 2022 👀](#automorph-2022-)
  - [Index](#index)
  - [Quickstart with Colab](#quickstart-with-colab)
  - [Install instruction for local/virtual machine](#install-instruction-for-localvirtual-machine)
    - [Requirements](#requirements)
    - [Package installation](#package-installation)
    - [Running](#running)
  - [Docker usage](#docker-usage)
  - [Common questions](#common-questions)
    - [Memory/ram error](#memoryram-error)
    - [Invalid results](#invalid-results)
  - [Components](#components)
  - [Citation](#citation)

&nbsp;



## Quickstart with Colab

Use the Google Colab and a free Tesla T4 gpu

[Colab link click](https://colab.research.google.com/drive/13Qh9umwRM1OMRiNLyILbpq3k9h55FjNZ?usp=sharing)


## Install instruction for local/virtual machine

### Requirements

1. Linux is preferred. For windows, install [MinGW-w64](https://www.mingw-w64.org/) for using commands below to set enviroment.
2. Anaconda or miniconda installed.
3. python=3.6, cudatoolkit=11.0, torch=1.7, etc. (installation steps below)
4. GPU is essential.


### Package installation

Step 1: create virtual environment:
```bash
conda update conda
conda create -n automorph python=3.6 -y
conda activate automorph
```

Step 2: install pytorch 1.7 and cudatoolkit 11.0
```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
```

Step 3: install other packages:
```bash
pip install --ignore-installed certifi
pip install -r requirement.txt
pip install efficientnet_pytorch
```

### Running

Activate virtual environment and clone the code.
```bash
conda activate automorph
git clone https://github.com/rmaphoh/AutoMorph.git
```

Put the images in folder 'images' and
```bash
sh run.sh
```


Please not that resolution_information.csv includes the resolution for image, i.e., size for each pixel. Please prepare it for the customised data in the same format.

&nbsp;

## Docker usage

Zero experience in Docker? No worries.

First, clone the github to <path/of/AutoMorph, e.g., /home/AutoMorph> and put the images in AutoMorph/images
```bash
git clone https://github.com/rmaphoh/AutoMorph.git
```

Then, pull our [docker image](https://hub.docker.com/r/yukundocker/image_automorph) and run the tool.
```bash
docker pull yukundocker/image_automorph
docker run  -v <path/of/AutoMorph>:/root/AutoMorph -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all yukundocker/image_automorph
source /root/set_up.sh
```


## Common questions

### Memory/ram error

We use Tesla T4 (16Gb) and 32vCPUs (120Gb). When you meet memory/ram issue in running, try to decrease batch size:

* ./M1_Retinal_Image_quality_EyePACS/test_outside.sh -b=64 to smaller, e.g., 32 or 16.
* ./M2_Artery_vein/test_outside.sh --batch-size=8 to smaller
* ./M2_lwnet_disc_cup/test_outside.sh --batchsize=8 to smaller


### Invalid results

In csv files, invalid values (e.g., optic disc segmentation failure) are indicated with -1.  


## Components

1. Vessel segmentation [BF-Net](https://github.com/rmaphoh/Learning-AVSegmentation.git)

2. Image pre-processing [EyeQ](https://github.com/HzFu/EyeQ.git) 

3. Optic disc segmentation [lwnet](https://github.com/agaldran/lwnet.git)

4. Feature measurement [retipy](https://github.com/alevalv/retipy.git)


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

