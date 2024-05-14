SRCDIR = src
OBJDIR = obj
BINDIR = bin

OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o encoding.o display_variables.o storage.o sparse_decoding.o GPU_decoding.o)
SOURCE  = $(addprefix $(SRCDIR)/, main.c decoding.c encoding.c display_variables.c storage.c sparse_decoding.c GPU_decoding.cu)
HEADER  = $(addprefix $(SRCDIR)/, decoding.h encoding.h defs.h display_variables.h storage.h sparse_decoding.h GPU_decoding.h) 
OUT     = $(BINDIR)/ldpc

CC      = gcc
FLAGS	= -g -c -Wall
NVCC 	= nvcc
CUFLAGS	= -std=c99 -O3 -m64 --gpu-architecture
MATH    = -lm

all: $(OUT)

$(OUT): $(OBJS) $(CUOBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS) $(MATH)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADER)
	$(CC) $(FLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(HEADER)
	$(NVCC) $(CUFLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(OUT)