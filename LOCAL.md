## Running on local/virtual machine

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
```

Step 2: Activate virtual environment and clone the code.
```bash
conda activate automorph
git clone https://github.com/rmaphoh/AutoMorph.git
cd AutoMorph
```

Step 3: install pytorch 1.7 and cudatoolkit 11.0
```bash
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -y
```

Step 4: install other packages:
```bash
pip install --ignore-installed certifi
pip install -r requirement.txt
pip install efficientnet_pytorch
```

### Running

Put the images in folder 'images' and
```bash
sh run.sh
```

Please note that resolution_information.csv includes the resolution for image, i.e., size for each pixel. Please prepare it for the customised data in the same format.



