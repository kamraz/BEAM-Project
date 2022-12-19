# BEAM Project

## Overview

## Implementation Details

## Results

Final Model
| Background | Adult | Juvenile | Chick | Food |
|------------|-------|----------|-------|------|
| 0.99       | 0.99  | 0.99     | 0.99  | 0.99 |

#### Successful Examples

#### Failed Examples

#### Future work

## Setup

#### Creating Google Cloud VM

- Open compute engine dash board
- Select marketplace and search for: "Deep Learning VM" or "Pytorch" select the google click to deploy ML vm and hit launch
- Sugested compute settings GPU: Nvidia T4, Machine Type: n1-standard-2
- Set framework to be: "PyTorch 1.12 (CUDA 11.3)"
- Select "Install NVIDIA GPU driver automatically on first startup?"
- Bootdisk as large as needed 150gb sugested to start

#### Adding SSH key to VM

- Open compute engine metadata page
- Click on ssh keys tab, add personal ssh key

#### On VM:
- Clone repo from github
- In repo run: `conda env create -f env.yml`
- Then `activate conda env`

## Useful Commands

#### Getting data out of a bucket

`gcloud storage cp -r BUCKET_NAME .`

#### Add swap (if model training fails try this)

In terminal:
```bash
sudo swapon --show
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon --show
```

## ARCHIVE:

## Creating new env

conda create --name NAME_HERE python=3.9

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pytorch-lightning
