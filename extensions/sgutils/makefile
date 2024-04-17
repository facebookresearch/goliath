export CUDA_VERSION:=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
export CPATH:=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include:${CPATH}
export LD_LIBRARY_PATH:=/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export PATH:=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
all:
	python setup.py build_ext --inplace