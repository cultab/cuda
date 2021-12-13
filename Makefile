
# TODO: architecture for each card

CUDA_SRC = radix.cu print.cu
#SRC = 
OBJ = $(CUDA_SRC:.cu=.o)

CC=nvcc
NVCC_FLAGS=-arch=sm_61 -forward-unknown-to-host-compiler
CCFLAGS=-Wall -Wextra -Wconversion

all: options debug

release: clean radix

debug: CCFLAGS += -DDEBUG -g
debug: radix

# default to debugging build
run: debug
	./radix

options:
	@echo build options:
	@echo "CC         = $(CC)"
	@echo "CCFLAGS    = $(CCFLAGS)"
	@echo "NVCC_FLAGS = $(NVCC_FLAGS)"
	@echo "CUDA_SRC   = $(CUDA_SRC)"
	@echo "OBJ        = $(OBJ)"

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CCFLAGS) -c $<

# dependencies
radix.o: print.h types.h
print.o: print.h
# types.o: types.h

radix: $(OBJ) $(CUDA_SRC)
	$(CC) -o $@ $(OBJ) $(NVCC_FLAGS) $(CCFLAGS) 

clean:
	rm -f radix $(OBJ)

.PHONY: all options clean run
