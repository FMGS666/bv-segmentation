# Installing miniconda
cd ~;
mkdir -p ~/miniconda3;
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh;
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3;
rm -rf ~/miniconda3/miniconda.sh;
# Initializing kaggle user
mkdir ~/.kaggle;
echo "{'username':'fmgsf12','key':'d1885b9fd9a3ed92a2a4fd70de6f5a7e'}" > ~/.kaggle/kaggle.json;
# Cloning the repository
