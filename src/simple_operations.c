#include "simple_operations.h"
#include <stdlib.h>

//=====DENSE=====
void d_mod2_vectmatmul(int* out,pchk mat,int* vect){
    for(int m=0; m<mat.n_row; m++){
        out[m]=0;
        for(int n=0; n<mat.n_col; n++){
            if(mat.A[m][n] == 1)
                out[m]^=vect[n];
        }
    }
}

void d_free_pchk(pchk mat){
    for(int i=0;i<mat.n_row;i++)
            free(mat.A[i]);
    free(mat.A);
}

//=====SPARSE=====

void  s_mod2_vectmatmul(int* out,pchk mat,int* vect){
    for(int r=0;r<mat.n_row;r++){
        for (int j = mat.A[1][r]; j <  mat.A[1][r+1]; j++)
            out[mat.A[0][j]] ^= vect[r];
    }
}

void s_free_pchk(pchk mat){
    free(mat.A[0]);
    free(mat.A[1]);
    free(mat.A);
}

//=====GENERAL=====

//this function performs the operation mat.vect =out
void mod2_vectmatmul(int* out,pchk mat,int* vect){
    switch(mat.type){
        case 0://dense
            d_mod2_vectmatmul(out,mat,vect);
            break;
        default://sparse
            s_mod2_vectmatmul(out,mat,vect);
    }
}

//this function frees the insides of the structure mat
void free_pchk(pchk mat){
    switch(mat.type){
        case 0://dense
            d_free_pchk(mat);
            break;
        default://sparse
            s_free_pchk(mat);
    }
}

//this function performs the operation c = a ^b vector wise
void bitwise_vectors(int *c, int *a, int *b, int size){
    for(int i=0;i<size;i++)
        c[i] = a[i] ^ b[i];
}