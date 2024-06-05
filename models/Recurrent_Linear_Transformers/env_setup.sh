# cuda
conda install nvidia/label/cuda-11.8::cuda
# jax
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# reqs
pip install -r requirements.txt
# headless rendering
conda install -c conda-forge mesalib
pip install PyOpenGL-accelerate