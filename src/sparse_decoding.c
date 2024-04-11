#include "sparse_decoding.h"
#include "display_variables.h"

// Function to check if it is a valid codeword
int scheck_codeword(pchk H, int *codeword){
    int check = 0;

    for (int i=0;i<H.n_row;i++){
        for (int j=H.A[1][i];j<H.A[1][i+1];j++)
            check ^= codeword[H.A[0][j]];
        if(check != 0)
            return 0;
    }
    return 1;

}

// Function to compute a BSC a priori probabilities
float* sbsc_a_priori_probabilities(int codeword_len,int *codeword)
{
    float *probabilities = (float*)malloc(codeword_len * sizeof(float));
    for (int i = 0; i < codeword_len; i++)
    {
        if (codeword[i] == 0)
        {
            probabilities[i] = log((1 - BSC_ERROR_RATE)/BSC_ERROR_RATE);
        }
        else
        {
            probabilities[i] = log(BSC_ERROR_RATE/(1 - BSC_ERROR_RATE));
        }
    }
    return probabilities;
}

// Function to return a priori probabilities based on the chosen mode
void sa_priori_probabilities(int mode,int codeword_len, int *codeword, float** probabilities)
{
    switch (mode)
    {
    case BSC_MODE:
        *probabilities = sbsc_a_priori_probabilities(codeword_len,codeword);
        break;
    default:
        break;
    }
}

//Function to compute extrinsic probabilities matrix (E)
void scompute_extrinsic(pchk H,pchk TH,float *M, float *E,float *LE, float *L,float *r, int *z){
    float p;

    for(int i=0;i<H.n_col;i++)
        L[i] = r[i];

    for (int j=0;j<H.n_row;j++){
        for (int i=H.A[1][j];i<H.A[1][j+1];i++){

            p = LE[j] / M[i];
            E[i] = log((1+p)/(1-p));

            L[ H.A[0][i]  ] += E[j];
        }
    }

    for(int i=0;i<H.n_col;i++)
        z[i] = (L[i] < 0) ? 1 : 0;
    
}

//Function to Update matrix M
void sUpdate_M(pchk H,pchk TH,float *M, float *E,float *LE, float *L){
    for (int j=0;j<H.n_row;j++){
        LE[j]=1;
        for (int i=H.A[1][j];i<H.A[1][j+1];i++){
            M[i] = L[ H.A[0][i] ] - E[i];

            LE[j] *= tanh(M[i]/2);
        }
    }
}

//function to innitialize the matrix M (and it's colapsed rows LE)
void Mi(pchk H, float *M, float *LE, float *r){

    for (int i=0; i < H.n_row; i++){
        LE[i]=1;
        for (int j=H.A[1][i];j<H.A[1][i+1];j++){
            M[j] = r[i];
            LE[i] *= tanh(M[i]/2);
        }
    }
}

// Function to decode the message
void sdecode(pchk H,pchk TH, int *recv_codeword, int *codeword_decoded){
    float *r;
    float *M,*E; //these are matrices in csr
    float *L,*LE;
    int try_n = 0;
    
    if (recv_codeword == NULL){
#ifdef DEBUG
        printf("Not a valid codeword\n");
#endif
        return ;
    }
    if(scheck_codeword(H, recv_codeword) == 1){
#ifdef DEBUG
        printf("valid codeword no errors detected\n");
#endif
        memcpy(codeword_decoded, recv_codeword, H.n_col * sizeof(int));
        return ;
    }
    // Initialize variables
    
    L  = (float *)calloc(H.n_col  , sizeof(float)); //colapsed E + prob column wise
    LE = (float *)calloc(H.n_row , sizeof(float)); //colapsed product of M rowwise
    M  = (float *)malloc(H.type   * sizeof(float)); //the index for M is H
    E  = (float *)malloc(H.type   * sizeof(float)); //the index for E is TH

    
    
#ifdef DEBUG
    printf("---------------------INITIALIZATIONS----------------------\n");
    
    printf("Codeword: ");
    print_vector_int(recv_codeword, H.n_col);
    printf("\n");
#endif
    
    //Compute LLR a priori probabilities (r)
    //this is the same has the "normal" decoding
    sa_priori_probabilities(CURR_MODE, H.n_col , recv_codeword, &r);

#ifdef DEBUG
    printf("Probabilities: ");
    print_vector_float(r, H.n_col);
#endif
    
    // Initialize matrix M
    Mi(H,M,LE,r);
    
#ifdef DEBUG
    printf("Matrix M: \n");
    print_sparse_float(H,M);
#endif
    while (try_n < MAX_ITERATIONS){
        try_n++;

#ifdef DEBUG
        printf("---------------------ITERATION %d----------------------\n", try_n);
#endif
        
        // Compute extrinsic probabilities matrix (E) it's compressed vector L and the best guess z
        scompute_extrinsic(H,TH,M,E,LE,L,r,codeword_decoded);
#ifdef DEBUG
        printf("Extrinsic probabilities: \n");
        print_sparse_float(H,E);

        printf("Final L: \n");
        print_vector_float(L, H.n_col);
#endif
        // Check if it is a valid codeword
        if (scheck_codeword(H, codeword_decoded) == 1){
#ifdef DEBUG
            printf("Completed after %d iterations\n", try_n);
#endif
            break;
        }
        // Update matrix M
        sUpdate_M(H,TH,M,E,LE,L);
    }
#ifdef DEBUG
    if(try_n == MAX_ITERATIONS)
        printf("Not completed after %d iterations\n", try_n);  
#endif
    
    free(M);
    free(E);
    free(r);
    free(L);
    free(LE);
    return ;
}