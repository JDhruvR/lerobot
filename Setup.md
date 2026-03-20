# Installation setup from - [https://huggingface.co/docs/lerobot/installation]
```bash  
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

conda create -y -n lerobot python=3.10

conda activate lerobot

conda install ffmpeg -c conda-forge

git clone https://github.com/huggingface/lerobot.git
cd lerobot

pip install -e .

pip install -e ".[smolvla]"

wandb login
```
