#ifndef GPU_LDPC
#define GPU_LDPC
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "display_variables.h"
#include "defs.h"

// Function to decode the message
void GPU_decode(pchk H, int *recv_codeword, int *codeword_decoded);