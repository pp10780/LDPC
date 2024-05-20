SRCDIR = src
OBJDIR = obj
BINDIR = bin

OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o simple_operations.o display_variables.o storage.o sparse_decoding.o simple_decoding.o)
SOURCE  = $(addprefix $(SRCDIR)/, main.c decoding.c simple_operations.c display_variables.c storage.c sparse_decoding.c simple_decoding.c)
HEADER  = $(addprefix $(SRCDIR)/, decoding.h simple_operations.h defs.h display_variables.h storage.h sparse_decoding.h simple_decoding.h) 
OUT     = $(BINDIR)/ldpc

CC      = gcc
FLAGS	= -std=c99 -g -c -Wall
NVCC 	= nvcc
CUFLAGS	= -O3 -m64 -c
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