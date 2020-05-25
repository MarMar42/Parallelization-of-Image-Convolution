INC="inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall
LFLAGS=-lglut -lGL

all:convolv

convolv: convolution.cu
	$(NVCC) $(NVCCFLAGS) -lineinfo convolution.cu -o convolv $(LFLAGS)

clean:
	rm convolv
