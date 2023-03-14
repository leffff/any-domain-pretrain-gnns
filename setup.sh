pip install -r requirements.txt
pip install -r testing_requirements.txt

pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric