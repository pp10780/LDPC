SRCDIR = src
OBJDIR = obj
BINDIR = bin
MATDIR = matrices

#remove GPU whilst testing
#LDPC
OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o encoding.o display_variables.o storage.o sparse_decoding.o )
SOURCE  = $(addprefix $(SRCDIR)/, main.c decoding.c encoding.c display_variables.c storage.c sparse_decoding.c )
HEADER  = $(addprefix $(SRCDIR)/, decoding.h encoding.h defs.h display_variables.h storage.h sparse_decoding.h ) 
OUT     = $(BINDIR)/ldpc

#GPU_decoding.cu GPU_decoding.h

#tests this didnt work remove this
TESTS 	= $(addprefix $(MATDIR)/, D_I20x20 D_I50x50 D_I100x100 D_I1000x1000 D_I10^4x10^4 )

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
	@./bin/ldpc matrices/D_I20x20 matrices/D_I20x20
	@./bin/ldpc matrices/D_I50x50 matrices/D_I50x50 
	@./bin/ldpc matrices/D_I100x100 matrices/D_I100x100 
	@./bin/ldpc matrices/D_I1000x1000 matrices/D_I1000x1000 
	@./bin/ldpc matrices/D_I10^4x10^4 matrices/D_I10^4x10^4 

clean:
	rm -f $(OBJDIR)/*.o
	rm -f $(BINDIR)/*