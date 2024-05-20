#include "simple_operations.h"

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


//=====SPARSE=====

void  s_mod2_vectmatmul(int* out,pchk mat,int* vect){
    for(int r=0;r<mat.n_row;r++){
        for (int j = mat.A[1][r]; j <  mat.A[1][r+1]; j++)
            out[mat.A[0][j]] ^= vect[r];
    }
}

//=====GENERAL=====

//this function perform the operation mat.vect =out
void mod2_vectmatmul(int* out,pchk mat,int* vect){
    switch(mat.type){
        case 0://dense
            d_mod2_vectmatmul(out,mat,vect);
            break;
        default://sparse
            s_mod2_vectmatmul(out,mat,vect);
    }
}

void bitwise_vectors(int *c, int *a, int *b, int size){
    for(int i=0;i<size;i++)
        c[i] = a[i] ^ b[i];
}