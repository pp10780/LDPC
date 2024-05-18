#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "display_variables.h"
#include "defs.h"

//function to bitwise xor a and b and store in c : c=a^b;
void bitwise_vectors(int *c, int *a, int *b, int size);

// Function to decode the message
void simple_decode(pchk H, int* recv_codeword, int* codeword_decoded);