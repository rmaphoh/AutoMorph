## Running on local/virtual machine

### Requirements

1. Linux or Mac are preferred. For windows, install [MinGW-w64](https://www.mingw-w64.org/) for using commands below to set enviroment.
2. Anaconda or miniconda installed.
3. python=3.11, torch=2.3, etc. (installation steps below)
4. GPU is essential -  NVIDIA (cuda) or M2 (mps).


### Package installation

Step 1: create virtual environment:
```bash
conda update conda
conda create -n automorph python=3.11 -y
```

Step 2: Activate virtual environment and clone the code.
```bash
conda activate automorph
git clone https://github.com/rmaphoh/AutoMorph.git
cd AutoMorph
```

Step 3: install pytorch 2.3.1
check CUDA version with ```nvcc --version```.
For CUDA cuda_12.1 run install 
```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Step 4: install other packages:
```bash
pip install --ignore-installed certifi
pip install -r requirement.txt
pip install efficientnet_pytorch==0.7.1 --no-deps 
```

### Running

Put the images in folder 'images' and
```bash
sh run.sh
```

Please note that resolution_information.csv includes the resolution for image, i.e., size for each pixel. Please prepare it for the customised data in the same format.



