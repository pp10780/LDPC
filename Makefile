SRCDIR = src
OBJDIR = obj
BINDIR = bin
MATDIR = matrices
CUDAPATH = /usr/local/cuda-12.4

#remove GPU whilst testing
#LDPC
OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o encoding.o display_variables.o storage.o sparse_decoding.o GPU_sparse_decoding.o)
SOURCE  = $(addprefix $(SRCDIR)/, main.cu decoding.cu encoding.cu display_variables.cu storage.cu sparse_decoding.c GPU_sparse_decoding.cu)
HEADER  = $(addprefix $(SRCDIR)/, decoding.h encoding.h defs.h display_variables.h storage.h sparse_decoding.h GPU_sparse_decoding.h) 
OUT     = $(BINDIR)/ldpc

CC      = gcc
FLAGS	= -std=c99 -g -c -Wall 
NVCC 	= $(CUDAPATH)/bin/nvcc
CUFLAGS	= --gpu-architecture=compute_61 -O3 -c
MATH    = -lm
LFLAGS = -L/$(CUDAPATH)/lib64 -lcudart

#COMPILING RULES
$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADER) 
	$(CC) $(FLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADER) 
	$(NVCC) $(CUFLAGS) $< -o $@

all: $(OUT)

ldpc: $(OUT)
$(OUT): $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS) $(MATH) 

test: 
	$(OUT) matrices/G1.csr matrices/H1.csr 0.1

tests: 
	@echo =====1000=====
	@echo rate 0.2
#	$(OUT) matrices/G1 tests/tests_1000_0.2_1
	$(OUT) tests/tests_1000_0.2_3G tests/tests_1000_0.2_3
	$(OUT) tests/tests_1000_0.2_5G tests/tests_1000_0.2_5
	$(OUT) tests/tests_1000_0.2_9G tests/tests_1000_0.2_9
	@echo rate 0.1
#	$(OUT) matrices/G1 tests/tests_1000_0.1_1
	$(OUT) tests/tests_1000_0.1_3G tests/tests_1000_0.1_3
	$(OUT) tests/tests_1000_0.1_5G tests/tests_1000_0.1_5
	$(OUT) tests/tests_1000_0.1_9G tests/tests_1000_0.1_9
	@echo rate 0.05
#	$(OUT) matrices/G1 tests/tests_1000_0.05_1
	$(OUT) tests/tests_1000_0.05_3G tests/tests_1000_0.05_3
	$(OUT) tests/tests_1000_0.05_5G tests/tests_1000_0.05_5
	$(OUT) tests/tests_1000_0.05_9G tests/tests_1000_0.05_9
	@echo rate 0.02
#	$(OUT) matrices/G1 tests/tests_1000_0.02_1
	$(OUT) tests/tests_1000_0.02_3G tests/tests_1000_0.02_3
	$(OUT) tests/tests_1000_0.02_5G tests/tests_1000_0.02_5
	$(OUT) tests/tests_1000_0.02_9G tests/tests_1000_0.02_9
	@echo rate 0.01
#	$(OUT) matrices/G1 tests/tests_1000_0.01_1
	$(OUT) tests/tests_1000_0.01_3G tests/tests_1000_0.01_3
	$(OUT) tests/tests_1000_0.01_5G tests/tests_1000_0.01_5
	$(OUT) tests/tests_1000_0.01_9G tests/tests_1000_0.01_9

break:
	@echo =====10000=====
	@echo rate 0.2
#	$(OUT) matrices/G1 tests/tests_10000_0.2_1
	$(OUT) tests/tests_10000_0.2_3G tests/tests_10000_0.2_3
	$(OUT) tests/tests_10000_0.2_5G tests/tests_10000_0.2_5
	$(OUT) tests/tests_10000_0.2_9G tests/tests_10000_0.2_9
	@echo rate 0.1
#	$(OUT) matrices/G1 tests/tests_10000_0.1_1
	$(OUT) tests/tests_10000_0.1_3G tests/tests_10000_0.1_3
	$(OUT) tests/tests_10000_0.1_5G tests/tests_10000_0.1_5
	$(OUT) tests/tests_10000_0.1_9G tests/tests_10000_0.1_9
	@echo rate 0.05
#	$(OUT) matrices/G1 tests/tests_10000_0.05_1
	$(OUT) tests/tests_10000_0.05_3G tests/tests_10000_0.05_3
	$(OUT) tests/tests_10000_0.05_5G tests/tests_10000_0.05_5
	$(OUT) tests/tests_10000_0.05_9G tests/tests_10000_0.05_9
	@echo rate 0.02
#	$(OUT) matrices/G1 tests/tests_10000_0.02_1
	$(OUT) tests/tests_10000_0.02_3G tests/tests_10000_0.02_3
	$(OUT) tests/tests_10000_0.02_5G tests/tests_10000_0.02_5
	$(OUT) tests/tests_10000_0.02_9G tests/tests_10000_0.02_9
	@echo rate 0.01
#	$(OUT) matrices/G1 tests/tests_10000_0.01_1
	$(OUT) tests/tests_10000_0.01_3G tests/tests_10000_0.01_3
	$(OUT) tests/tests_10000_0.01_5G tests/tests_10000_0.01_5
	$(OUT) tests/tests_10000_0.01_9G tests/tests_10000_0.01_9
	@echo =====100000=====
	@echo rate 0.2
#	$(OUT) matrices/G1 tests/tests_100000_0.2_1
	$(OUT) matrices/G1 tests/tests_100000_0.2_3
	$(OUT) matrices/G1 tests/tests_100000_0.2_5
	$(OUT) matrices/G1 tests/tests_100000_0.2_9
	@echo rate 0.1
#	$(OUT) matrices/G1 tests/tests_100000_0.1_1
	$(OUT) matrices/G1 tests/tests_100000_0.1_3
	$(OUT) matrices/G1 tests/tests_100000_0.1_5
	$(OUT) matrices/G1 tests/tests_100000_0.1_9
	@echo rate 0.05
#	$(OUT) matrices/G1 tests/tests_100000_0.05_1
	$(OUT) matrices/G1 tests/tests_100000_0.05_3
	$(OUT) matrices/G1 tests/tests_100000_0.05_5
	$(OUT) matrices/G1 tests/tests_100000_0.05_9
	@echo rate 0.02
#	$(OUT) matrices/G1 tests/tests_100000_0.02_1
	$(OUT) matrices/G1 tests/tests_100000_0.02_3
	$(OUT) matrices/G1 tests/tests_100000_0.02_5
	$(OUT) matrices/G1 tests/tests_100000_0.02_9
	@echo rate 0.01
#	$(OUT) matrices/G1 tests/tests_100000_0.01_1
	$(OUT) matrices/G1 tests/tests_100000_0.01_3
	$(OUT) matrices/G1 tests/tests_100000_0.01_5
	$(OUT) matrices/G1 tests/tests_100000_0.01_9
clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(BINDIR)/*