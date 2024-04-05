#include "sparse_decoding.h"
#include "display_variables.h"

// Function to check if it is a valid codeword
int scheck_codeword(pchk H, int *codeword){
    int check = 0;

    for (int i=0;i<H.n_row;i++){
        for (int j=0;j<H.A[i][0];j++)
            check ^= codeword[H.A[i][j+1]];
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
void scompute_extrinsic(pchk H,pchk TH,float ** E, float **M, float *LE){
    float p;
    int mj;

    //compute new LE
    for (int i=0;i<TH.n_row;i++){
        LE[i]=1;
        for (int j=0; j<TH.A[i][0]; j++)
            LE[i] *= tanh(M[i][j]/2);
    }

    //Update E
    for (int j=0;j<H.n_row;j++){
        for (int i=0;i<H.A[j][0];i++){
            //finding the real index where Mi,j is stored
            for(mj=0;TH.A[ H.A[j][i+1] ][mj+1] != j;mj++);
            p = LE[j] / M[i][mj];
            E[j][i] = log((1+p)/(1-p));
        }
    }
}

//Function to Update matrix M
void sUpdate_M(pchk H,pchk TH,float **M,float *L,float **E){
    int mi;
    
    for (int i=0;i<TH.n_row;i++){
        for (int j=0;j<TH.A[i][0];j++){
            //finding the real index where Ei,j is stored
            for(mi=0;H.A[ TH.A[i][j+1] ][mi]!=i;mi++);

            M[i][j] = L[i] - E[j][mi];
        }
    }
}

//Function to get new best guess as well has probabilities
void sGet_state(pchk H, float *L, int *z, float *r, float **E){

    for (int j=0;j<H.n_row;j++){
        L[j] = r[j];
        for (int i=0;i<H.A[j][0];i++)
            L[j] += E[j][i];
        z[j] = (L[j] < 0) ? 1 : 0;
    }
}

// Function to decode the message
void sdecode(pchk H,pchk TH, int *recv_codeword, int *codeword_decoded){
    float *r;
    float **M,**E;
    float *L,*LE;
    int try_n = 0;
    
    if (recv_codeword == NULL){
#ifdef DEBUG
        printf("Not a valid codeword\n");
#endif
        return ;
    }
    if(scheck_codeword(H, recv_codeword) == 1){
        memcpy(codeword_decoded, recv_codeword, H.n_col * sizeof(int));
        return ;
    }
    // Initialize variables
    //H is the index for E and TH is the index for M
    L = (float *)calloc(H.n_row , sizeof(float*)); //colapsed E + prob
    LE = (float *)calloc(TH.n_row , sizeof(float*)); //colapsed product of M

    M = (float**)malloc(TH.n_row * sizeof(float*));
    for(int i=0; i<TH.n_row ; i++){
        if(TH.A[i][0] != 0)
            M[i]=(float *)malloc(TH.A[i][0] * sizeof(float*));
    }

    E = (float**)malloc(H.n_col * sizeof(float*));
    for(int j=0; j<H.n_row ; j++){
        if(H.A[j][0] != 0)
            E[j]=(float *)malloc(H.A[j][0] * sizeof(float*));
    }
    
#ifdef DEBUG
    printf("---------------------INITIALIZATIONS----------------------\n");
    
    printf("Codeword: ");
    print_vector_int(recv_codeword, H.n_col);
    printf("\n");
#endif

    //Compute LLR a priori probabilities (r)
    a_priori_probabilities(CURR_MODE, H.n_col , recv_codeword, &r);

#ifdef DEBUG
    printf("Probabilities: ");
    print_vector_float(r, H.n_col);
#endif
    
    // Initialize matrix M
    for (int i = 0; i < TH.n_row; i++){
        for (int j = 0; j < TH.A[i][0]; j++)
            M[i][j] = r[i];
    }
#ifdef DEBUG
    printf("Matrix M: \n");
    print_sparse_float(TH,M);
#endif

    while (try_n < MAX_ITERATIONS){
        try_n++;

#ifdef DEBUG
        printf("---------------------ITERATION %d----------------------\n", try_n);
#endif

        // Compute extrinsic probabilities matrix (E)
        scompute_extrinsic(H,TH,E,M,LE);
#ifdef DEBUG
        printf("Extrinsic probabilities: \n");
        print_sparse_float(H,E);
#endif
        // Test
        sGet_state(H,L,codeword_decoded,r,E);
#ifdef DEBUG
        printf("Final L: ");
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
        sUpdate_M(H,TH,M,L,E);
    }
#ifdef DEBUG
    if(try_n == MAX_ITERATIONS)
        printf("Not completed after %d iterations\n", try_n);  
#endif
    for (int i = 0; i < H.n_row; i++){
        free(M[i]);
        free(E[i]);
    }
    free(M);
    free(E);
    free(r);
    free(L);
    free(LE);
    return ;
}