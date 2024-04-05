#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "display_variables.h"

// Function to decode the message using sparse matrix
void sdecode(pchk H,pchk TH, int *recv_codeword, int *codeword_decoded);