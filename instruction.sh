##############################################
# install certifi
pip install --ignore-installed certifi==2021.5.30

###############################################
# if you prefer to use conda 
# create a virtual enviroment 'automorph'
# install cuda 11.0 and pytorch 1.7
conda create -n automorph python=3.6
conda activate automorph
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# OR if you has cuda 11.0 already and prefer to use pip
# install torch 1.7
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
###############################################


# install all other packages
pip install -r requirement.txt
pip install efficientnet_pytorch

