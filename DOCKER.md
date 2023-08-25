## Running with Docker

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



