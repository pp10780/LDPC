SRCDIR = src
OBJDIR = obj
BINDIR = bin
MATDIR = matrices

#remove GPU whilst testing
#LDPC
OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o encoding.o display_variables.o storage.o sparse_decoding.o GPU_decoding.o)
SOURCE  = $(addprefix $(SRCDIR)/, main.c decoding.c encoding.c display_variables.c storage.c sparse_decoding.c GPU_decoding.cu)
HEADER  = $(addprefix $(SRCDIR)/, decoding.h encoding.h defs.h display_variables.h storage.h sparse_decoding.h GPU_decoding.h) 
OUT     = $(BINDIR)/ldpc

#GPU_decoding.cu GPU_decoding.h

CC      = gcc
FLAGS	= -std=c99 -g -c -Wall
NVCC 	= nvcc
CUFLAGS	= -O3 -m64 -c
MATH    = -lm

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
	./bin/ldpc matrices/G1 matrices/H1

all_tests: 
	@./bin/ldpc matrices/G1 matrices/H1
	
clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(BINDIR)/*