
# TODO: architecture for each card
all:
	nvcc -o radix radix.cu -arch=sm_61 # -forward-unknown-to-host-compiler -Wall
