#include "encoding.h"


// Function to encode the key
void encode(int *key, pchk generator, int *message)
{
    if(generator.type == 0){
        //normal
        for(int row = 0; row < generator.n_row; row++){
            for(int col = 0; col < generator.n_col; col++){
                if(generator.A[row][col]==1)
                    message[row] ^= key[col];
            }
        }
    }
    else{
        //sparse
        for(int r=0;r<generator.n_row;r++){
            for (int j = generator.A[1][r]; j <  generator.A[1][r+1]; j++)
                message[generator.A[0][j]] ^= key[j];
        }
            
    }

}