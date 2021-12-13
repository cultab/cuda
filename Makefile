
# TODO: architecture for each card

SRC = radix.cu print_array.cu
OBJ = $(SRC:.c=.o)

NVCC_FLAGS=-arch=sm_61
CFLAGS=-Wall -forward-unknown-to-host-compiler

all: options radix

radix.o: print_array.h types.h
print_array.o: print_array.h

options:
	@echo build options:
	@echo "CFLAGS     = $(CFLAGS)"
	@echo "NVCC_FLAGS = $(NVCC_FLAGS)"

radix: $(OBJ)
	nvcc -o $@ $(OBJ) $(NVCC_FLAGS) $(CFLAGS) 

clean:
	rm -f radix $(OBJ)

.PHONY: all options clean
