#include "encoding.h"


// Function to encode the message
void encode(int *message, pchk generator, int *codeword)
{
    if(generator.type == 0){
        //normal
        for(int k = 0; k < generator.n_row; k++){
            if(message[k] == 1)
            {
                for(int n = 0; n < generator.n_col; n++)
                {
                    codeword[n] ^= generator.A[k][n];
                }
            }
        }
    }
    else{
        //sparse
        for(int r=0;r<generator.n_row;r++){
            for(int c=0;c<generator.A[r][0];c++){
                codeword[generator.A[r][c+1]] ^= message[r];
            }
        }
    }

}