#ifndef DECODING_H
#define DECODING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "display_variables.h"
#include "defs.h"

void decode(pchk H, int* recv_codeword, int* codeword_decoded, float error_rate);

#endif