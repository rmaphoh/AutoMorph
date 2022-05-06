
# install certifi
pip install --ignore-installed certifi==2021.5.30

# need cuda 11.0 - can be done by conda install cudatoolkit
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# OR if you has cuda 11.0 already
# install torch 1.7 and torchvision 0.8
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


# install all other packages
pip install -r requirement.txt


