# eagles

## Getting data out of a bucket

gcloud storage cp -r BUCKET_NAME .

## Creating new env

conda create --name NAME_HERE python=3.9

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pytorch-lightning

## Add swap

sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
