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
            for (int j = generator.A[1][r]; j <  generator.A[1][r+1]; j++)
                codeword[generator.A[0][j]] ^= message[r];
        }
            
    }

}