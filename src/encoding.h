#ifndef ENCODING_H
#define ENCODING_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "display_variables.h"
#include "defs.h"

// Function to encode the message
void encode(int* message, pchk generator, int* codeword);

#endif