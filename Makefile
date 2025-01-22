SRCDIR = src
OBJDIR = obj
BINDIR = bin
MATDIR = matrices
CUDAPATH = 

#remove GPU whilst testing
#LDPC
OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o encoding.o display_variables.o storage.o sparse_decoding.o )
SOURCE  = $(addprefix $(SRCDIR)/, main.c decoding.c encoding.c display_variables.c storage.c sparse_decoding.c )
HEADER  = $(addprefix $(SRCDIR)/, decoding.h encoding.h defs.h display_variables.h storage.h sparse_decoding.h ) 
OUT     = $(BINDIR)/ldpc

#GPU_decoding.cu GPU_decoding.h

CC      = gcc
FLAGS	= -std=c99 -g -c -Wall
NVCC 	= nvcc
CUFLAGS	= -O3 -m64 --gpu-architecture compute_61
MATH    = -lm
CUCCFLAGS = $(CUDAPATH) -lcuda

#COMPILING RULES
$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADER) 
	$(CC) $(FLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADER)
	$(NVCC) $(CUFLAGS) $< -o $@

all: $(OUT)

ldpc: $(OUT)
$(OUT): $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS) $(MATH) $(CUCCFLAGS)

gpu: 
	nvcc -O3 -m64 --gpu-architecture compute_61 src/GPU_sparse_decoding.cu -o bin/GPU_sparse
test: 
	./bin/GPU_sparse matrices/G1 matrices/H1 0.1

tests: 
	@echo =====1000=====
	@echo rate 0.2
#	./bin/GPU_sparse matrices/G1 tests/tests_1000_0.2_1
	./bin/GPU_sparse tests/tests_1000_0.2_3G tests/tests_1000_0.2_3
	./bin/GPU_sparse tests/tests_1000_0.2_5G tests/tests_1000_0.2_5
	./bin/GPU_sparse tests/tests_1000_0.2_9G tests/tests_1000_0.2_9
	@echo rate 0.1
#	./bin/GPU_sparse matrices/G1 tests/tests_1000_0.1_1
	./bin/GPU_sparse tests/tests_1000_0.1_3G tests/tests_1000_0.1_3
	./bin/GPU_sparse tests/tests_1000_0.1_5G tests/tests_1000_0.1_5
	./bin/GPU_sparse tests/tests_1000_0.1_9G tests/tests_1000_0.1_9
	@echo rate 0.05
#	./bin/GPU_sparse matrices/G1 tests/tests_1000_0.05_1
	./bin/GPU_sparse tests/tests_1000_0.05_3G tests/tests_1000_0.05_3
	./bin/GPU_sparse tests/tests_1000_0.05_5G tests/tests_1000_0.05_5
	./bin/GPU_sparse tests/tests_1000_0.05_9G tests/tests_1000_0.05_9
	@echo rate 0.02
#	./bin/GPU_sparse matrices/G1 tests/tests_1000_0.02_1
	./bin/GPU_sparse tests/tests_1000_0.02_3G tests/tests_1000_0.02_3
	./bin/GPU_sparse tests/tests_1000_0.02_5G tests/tests_1000_0.02_5
	./bin/GPU_sparse tests/tests_1000_0.02_9G tests/tests_1000_0.02_9
	@echo rate 0.01
#	./bin/GPU_sparse matrices/G1 tests/tests_1000_0.01_1
	./bin/GPU_sparse tests/tests_1000_0.01_3G tests/tests_1000_0.01_3
	./bin/GPU_sparse tests/tests_1000_0.01_5G tests/tests_1000_0.01_5
	./bin/GPU_sparse tests/tests_1000_0.01_9G tests/tests_1000_0.01_9

break:
	@echo =====10000=====
	@echo rate 0.2
#	./bin/GPU_sparse matrices/G1 tests/tests_10000_0.2_1
	./bin/GPU_sparse tests/tests_10000_0.2_3G tests/tests_10000_0.2_3
	./bin/GPU_sparse tests/tests_10000_0.2_5G tests/tests_10000_0.2_5
	./bin/GPU_sparse tests/tests_10000_0.2_9G tests/tests_10000_0.2_9
	@echo rate 0.1
#	./bin/GPU_sparse matrices/G1 tests/tests_10000_0.1_1
	./bin/GPU_sparse tests/tests_10000_0.1_3G tests/tests_10000_0.1_3
	./bin/GPU_sparse tests/tests_10000_0.1_5G tests/tests_10000_0.1_5
	./bin/GPU_sparse tests/tests_10000_0.1_9G tests/tests_10000_0.1_9
	@echo rate 0.05
#	./bin/GPU_sparse matrices/G1 tests/tests_10000_0.05_1
	./bin/GPU_sparse tests/tests_10000_0.05_3G tests/tests_10000_0.05_3
	./bin/GPU_sparse tests/tests_10000_0.05_5G tests/tests_10000_0.05_5
	./bin/GPU_sparse tests/tests_10000_0.05_9G tests/tests_10000_0.05_9
	@echo rate 0.02
#	./bin/GPU_sparse matrices/G1 tests/tests_10000_0.02_1
	./bin/GPU_sparse tests/tests_10000_0.02_3G tests/tests_10000_0.02_3
	./bin/GPU_sparse tests/tests_10000_0.02_5G tests/tests_10000_0.02_5
	./bin/GPU_sparse tests/tests_10000_0.02_9G tests/tests_10000_0.02_9
	@echo rate 0.01
#	./bin/GPU_sparse matrices/G1 tests/tests_10000_0.01_1
	./bin/GPU_sparse tests/tests_10000_0.01_3G tests/tests_10000_0.01_3
	./bin/GPU_sparse tests/tests_10000_0.01_5G tests/tests_10000_0.01_5
	./bin/GPU_sparse tests/tests_10000_0.01_9G tests/tests_10000_0.01_9
	@echo =====100000=====
	@echo rate 0.2
#	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.2_1
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.2_3
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.2_5
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.2_9
	@echo rate 0.1
#	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.1_1
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.1_3
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.1_5
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.1_9
	@echo rate 0.05
#	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.05_1
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.05_3
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.05_5
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.05_9
	@echo rate 0.02
#	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.02_1
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.02_3
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.02_5
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.02_9
	@echo rate 0.01
#	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.01_1
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.01_3
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.01_5
	./bin/GPU_sparse matrices/G1 tests/tests_100000_0.01_9
clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(BINDIR)/*

gpu_dense:
	nvcc -O3 -m64 --gpu-architecture compute_61 src/GPU_decoding.cu -o bin/GPU_dense

gpu_sparse:
	nvcc -O3 -m64 --gpu-architecture compute_61 src/GPU_sparse_decoding.cu -o bin/GPU_sparse
