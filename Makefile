
# TODO: architecture for each card

PROGRAMS = radix bitonic
CUDA_SRC = radix.cu print.cu bitonic.cu
# OBJ = $(CUDA_SRC:.cu=.o)
OBJ = print.o

CC=nvcc
NVCC_FLAGS=-arch=sm_61 -forward-unknown-to-host-compiler
NVCC_COMPILE_ONLY_FLAGS=--device-c  # relocatable device code
CCFLAGS=-Wall -Wextra -Wconversion

#CCACHE := $(shell command -v ccache 2> /dev/null)

# default to debugging build
all: options debug

release: clean radix bitonic

profile: release
	nvprof ./radix

debug: CCFLAGS += -DDEBUG -g
debug: $(PROGRAMS)

run_r: debug
	./radix

run_b: debug
	./bitonic

options:
	@echo build options:
	@echo "CC          = $(CC)"
	@echo "CCACHE      = $(CCACHE)"
	@echo "CCFLAGS     = $(CCFLAGS)"
	@echo "NVCC_FLAGS  = $(NVCC_FLAGS)"
	@echo "CUDA_SRC    = $(CUDA_SRC)"
	@echo "OBJ         = $(OBJ)"

# compile
%.o: %.cu
	$(CCACHE) $(CC) $(NVCC_FLAGS) $(NVCC_COMPILE_ONLY_FLAGS)  $(CCFLAGS) -c $<

# dependencies
radix.o: print.cuh types.cuh
bitonic.o: print.cuh types.cuh
print.o: print.cuh

# link  HACK: hardcoding objects
$(PROGRAMS): bitonic.o radix.o $(OBJ) $(CUDA_SRC)
	$(CCACHE) $(CC) -o $@ $@.o $(OBJ) $(NVCC_FLAGS) $(CCFLAGS)


clean:
	rm -f radix bitonic $(OBJ)

.PHONY: all options clean run debug release profile
