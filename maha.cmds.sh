


# conda create -n dl python=3.10
# conda activate dl
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install numpy pandas scikit-learn matplotlib seaborn jupyter notebook tqdm


conda create --name mygpt python==3.10 -y
conda activate mygpt
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name mygpt --display-name "mygpt"
