
# TODO: architecture for each card

# PROGRAMS = radix bitonic
CUDA_SRC = main.cu radix.cu print.cu #bitonic.cu
OBJ = $(CUDA_SRC:.cu=.o)
# OBJ = print.o

CC=nvcc
NVCC_FLAGS=-arch=sm_61 -forward-unknown-to-host-compiler
NVCC_COMPILE_ONLY_FLAGS=--device-c  # relocatable device code
CCFLAGS=-Wall -Wextra -Wconversion -fopenmp

#CCACHE := $(shell command -v ccache 2> /dev/null)

# default to debugging build
all: options debug

release: clean main

profile: release
	nvprof ./radix

debug: CCFLAGS += -DDEBUG -g
debug: main

run: debug
	main

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
main.o: radix.h bitonic.h print.h types.h
radix.o: print.h types.h
# bitonic.o: print.h types.h
print.o: print.h

main: $(OBJ)
	$(CCACHE) $(CC) -o $@ $(OBJ) $(NVCC_FLAGS) $(CCFLAGS)

clean:
	rm -f main $(OBJ)

.PHONY: all options clean run debug release profile
