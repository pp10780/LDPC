#include "simple_decoding.h"

// mod2 vector multiplication mat*vect = out
void mod2_vectmatmul(int* out,pchk mat,int* vect){
    for(int m=0; m<mat.n_row; m++){
        out[m]=0;
        for(int n=0; n<mat.n_col; n++){
            if(mat.A[m][n] == 1)
                out[m]^=vect[n];
        }
    }
}

//swap two interger values from their position
void swap(int* a,int *b){
    int c=*a;
    *a=*b;
    *b=c;
}

//swap two interger values from their position for two vectors
void swap_vectors(int *a, int *b, int size){
    for(int i=0;i<size;i++)
        swap(&(a[i]),&(b[i]));
}

//c = a ^ b but for intire vectors
void bitwise_vectors(int *c, int *a, int *b, int size){
    for(int i=0;i<size;i++)
        c[i] = a[i] ^ b[i];
}

//function to obtain linear solve function(out) for in,
//this may create a function that's not the inverse for non square matrices
//out has the same dimensions of in but if the matrix is not square it will always have a part that's all 0
void linear_solve_function(pchk* out,pchk in){
    int srow;

    if(in.n_row>in.n_col){
        printf("Cannot put identity matrix\n");
        exit(-1);
    }

    //this matrix will initiated with the same values has in in order to not change those values
    int **mat=(int **) malloc(in.n_row*sizeof(int *));
    for(int row=0;row<in.n_row;row++){
        mat[row]=(int *) malloc(in.n_col*sizeof(int));
        for(int col=0;col<in.n_col;col++)
            mat[row][col]=in.A[row][col];
    }

    //innitialize out with the required size and identity at the start
    out->n_row = in.n_row;
    out->n_col = in.n_col;
    out->A = (int **) malloc(out->n_row*sizeof(int *));
    
    for(int row=0; row<out->n_row; row++){
        out->A[row] = (int *) calloc(out->n_col,sizeof(int));
        //identity part
        out->A[row][row]=1;
    }

    //Gaussian elimination going down
    //all gaussian elimination operation will also be done to out which will start has a identity matrix
    for(int row=0; row<in.n_row-1; row++){
        //find row that has a '1' in the next position for 
        for(srow=row; mat[srow][row]!=1 ;srow++){
            if(srow==in.n_row){
                printf("linear solve didn't work H is not lineary independent!\n");
                exit(-1);
            }
        }
        //swap rows if the "current" one doesn't have a '1' in the column
        if(srow != row){
            swap_vectors(mat[row]   , mat[srow]   , in.n_col);
            swap_vectors(out->A[row], out->A[srow], in.n_col);
        }
        //bitwise and all vectors that have a '1' in the column
        for(srow = row+1;srow<in.n_row;srow++){
            if(mat[srow][row] == 1){
                bitwise_vectors(mat[srow]    , mat[row]   , mat[srow]   , in.n_col);
                bitwise_vectors(out->A[srow] , out->A[row], out->A[srow], in.n_col);
            }
        }
    }
    if(mat[in.n_row-1][in.n_row-1]!=1){
        printf("linear solve didn't work H is not lineary independent!\n");
        exit(-1);
    }

    //Gaussian elimination going up
    for(int row=in.n_row-1; row > -1; row--){
        //no need to check if there's a '1' in the column has that has already been done
        
        //bitwise and all vectors that have a '1' in the column
        for(srow = row-1;srow > -1;srow--){
            if(mat[srow][row] == 1){
                bitwise_vectors(mat[srow]    , mat[row]   , mat[srow]   , in.n_col);
                bitwise_vectors(out->A[srow] , out->A[row], out->A[srow], in.n_col);
            }
        }
    }

    for(int row=0;row<in.n_row;row++)
        free(mat[row]);
    free(mat);
}

void simple_decode(pchk H, int* recv_codeword, int* codeword_decoded){
    int *s,*e;
    pchk D;
    D.type=0;
    
    //allocate variables
    //sindrome is expanded from the size of H.n_col to n_row by adding 0s at the end
    //  I don't really thik it's necessary to do this part if I calculate D beforehand
    s = (int *) calloc(H.n_row , sizeof(int));
    e = (int *) calloc(H.n_row , sizeof(int));

    //get sindrome -> matrix vector multiplication with mod 2 Hc' = s
    mod2_vectmatmul(s,H,recv_codeword);
#ifdef DEBUG
    printf("s:\n");
    print_vector_int(s,H.n_row );
    printf("\n");
#endif
    //linear solve He = s to find e
    //this math can be done apriori has He = Is <=> Ie = e = Ds
    //where D can be calculated before and is the inver of H expanded with 0s ???
    //I am putting the math here anyway because I don't know where else to put it
    linear_solve_function(&D,H);
#ifdef DEBUG
    printf("D:\n");
    print_parity_check(D);
    printf("\n");
#endif

    //get error from sindrome
    mod2_vectmatmul(e,D,s);
 #ifdef DEBUG
    printf("e:\n");
    print_vector_int(e,H.n_row );
    printf("\n");
#endif

    //remove error from message
    bitwise_vectors(codeword_decoded,recv_codeword,e,H.n_row);

}