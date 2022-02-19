
# TODO: architecture for each card

NAME = sort

#cliff :) \ o
CUDA_SRC = \
	main.cu \
	print.cu \
	radix.cu  \
	helpers.cu \
	bitonic.cu  \
	counting.cu  \
	prefix_sum.cu

OBJ = $(CUDA_SRC:.cu=.o)

CC=nvcc
NVCC_FLAGS=-arch=sm_61 -forward-unknown-to-host-compiler
CCFLAGS=-Wall -Wextra -Wconversion #-fopenmp
# relocatable device code
NVCC_COMPILE_ONLY_FLAGS=--device-c

CCACHE := $(shell command -v ccache 2> /dev/null)

# default to debugging build
all: options debug

release: CCFLAGS += -O3
release: options clean
	+make main

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
	$(CCACHE) $(CC) $(NVCC_FLAGS) $(NVCC_COMPILE_ONLY_FLAGS) $(CCFLAGS) -c $<

# dependencies
main.o: radix.h bitonic.h print.h types.h helpers.h
radix.o: print.h types.h helpers.h
bitonic.o: print.h types.h helpers.h
print.o: print.h helpers.h

# link
main: $(OBJ)
	$(CCACHE) $(CC) $(OBJ) $(NVCC_FLAGS) $(CCFLAGS) -o $(NAME)

clean:
	rm -f main $(OBJ)

render:
	R --quiet -e "require(rmarkdown);render('report.rmd');"

submit:
	cp report.pdf 171014.pdf
	zip 171014.zip 171014.pdf

.PHONY: all options clean run debug release profile render submit
