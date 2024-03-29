SRCDIR = src
OBJDIR = obj
BINDIR = bin

OBJS    = $(addprefix $(OBJDIR)/, main.o decoding.o encoding.o display_variables.o)
SOURCE  = $(addprefix $(SRCDIR)/, main.c decoding.c encoding.c display_variables.c)
HEADER  = $(addprefix $(SRCDIR)/, decoding.h encoding.h defs.h display_variables.h)
OUT     = $(BINDIR)/ldpc
CC      = gcc
FLAGS   = -g -c -Wall
MATH    = -lm

all: $(OUT)

$(OUT): $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS) $(MATH)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(HEADER)
	$(CC) $(FLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(OUT)