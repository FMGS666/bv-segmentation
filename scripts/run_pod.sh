# Installing miniconda
cd ~;
mkdir -p ~/miniconda3;
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh;
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3;
rm -rf ~/miniconda3/miniconda.sh;
alias conda="~/miniconda/bin/conda";
# Initializing kaggle user
mkdir ~/.kaggle;
echo "{'username':'fmgsf12','key':'d1885b9fd9a3ed92a2a4fd70de6f5a7e'}" > ~/.kaggle/kaggle.json;
# Pulling the code from the internet
mkdir ~/blood-vessel-seg;
cd ~/blood-vessel-seg;
wget https://fmgs666.github.io/FGMS666.github.io/bv-seg.tar.gz;
wget https://fmgs666.github.io/FGMS666.github.io/env.yml;
tar -xvzf ./bv-seg.tar.gz;
rm bv-seg.tar.gz;
# Creating folder structure
mkdir data;
mkdir logs;
mkdir models;
mkdir models/pretrained;
# Downloading pre-trained weights
wget -O models/pretrained/model_swinvit.pt https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt;
# Creating conda environment
conda env create -f env.yml;
# Downloading the data using the kaggle api
conda run -n blood-vessel-seg kaggle competitions download -c blood-vessel-segmentation;
unzip blood-vessel-segmentation.zip -d data;
rm blood-vessel-segmentation.zip;
# Creating volumes 
conda run -n python3 -m bv-seg sample --n-samples 5 --context-length 100;