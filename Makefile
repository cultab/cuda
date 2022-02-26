
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

convert:
	R --quiet -e "require(rmarkdown);render('report.md.rmd');"
	mv report.md.md Readme.md

submit:
	cp report.pdf Project_Παράλληλου_Υπολογισμού.pdf
	7z a Project_Παράλληλου_Υπολογισμού_171014.zip Project_Παράλληλου_Υπολογισμού.pdf \
		 graph.R bench.sh Makefile bitonic.cu counting.cu helpers.cu main.cu prefix_sum.cu \
		 print.cu radix.cu bitonic.h counting.h helpers.h prefix_sum.h print.h \
		 radix.h types.h results_1.csv results_2.csv

.PHONY: all options clean run debug release profile render submit
