#include "decoding.h"
#include <time.h>

// Function to check if it is a valid codeword
int check_codeword(pchk H, int *codeword)
{
    int check = 0;

    for (int m = 0; m < H.n_row; m++)
    {
        for(int n = 0; n < H.n_col; n++)
        {
            if (codeword[n] == 1)
            {
                check ^= H.A[m][n];
            }
        }
        if (check == 1)
            return 0;
    }

    return 1;

}

// Function to compute a BSC a priori probabilities
float* bsc_a_priori_probabilities(int codeword_len,int *codeword)
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
void a_priori_probabilities(int mode,int codeword_len, int* codeword, float** probabilities)
{
    switch (mode)
    {
    case BSC_MODE:
        *probabilities = bsc_a_priori_probabilities(codeword_len,codeword);
        break;
    default:
        break;
    }
}

//Function to compute initial state of M after obtaining a priori probabilities
void Initial_state(pchk H, float** M, float* probabilities){
    for (int m = 0; m < H.n_row; m++)
    {
        for (int n = 0; n < H.n_col; n++)
        {
            if (H.A[m][n] == 0)
                M[m][n] = 0;
            else
                M[m][n] = probabilities[n];
        }
    }
    return;
}

// Function to compute the product of the other probabilities
float product(float** M, int m, int n,int len)
{
    float product = 1;
    for (int i = 0; i < len; i++)
    {
        if (i != n && M[m][i] != 0)
        {
            product *= tanh(M[m][i]/2);
        }
    }
    return product;
}


//Function to compute extrinsic probabilities matrix (E)
void compute_extrinsic(pchk H,float ** extrinsic_probabilities, float **M){
    float product_value = 0;
    for (int m = 0; m < H.n_row ; m++){
        for (int n = 0; n < H.n_col ; n++){
            if (H.A[m][n] == 0){
                extrinsic_probabilities[m][n] = 0;
            } else {
                product_value = product(M, m, n,H.n_col);
                extrinsic_probabilities[m][n] = log((1+product_value)/(1-product_value));
            }
        }
    }
}

//Function to Update matrix M
void Update_M(pchk H,float **M,float *L,float **extrinsic_probabilities){
    for (int n = 0; n < H.n_col; n++)
    {
        for (int m = 0; m < H.n_row; m++)
        {
            if (H.A[m][n] == 0)
            {
                M[m][n] = 0;
            }
            else
            {
                M[m][n] = L[n] - extrinsic_probabilities[m][n];
            }
            
        }
    }
}

//Function to get new best guess as well has probabilities
void Get_state(pchk H, float *L, int *codeword_decoded, float *probabilities, float **extrinsic_probabilities){
    for(int n = 0; n < H.n_col; n++)
    {
        L[n] = probabilities[n];
        for(int m = 0; m < H.n_row; m++)
        {
            L[n] += extrinsic_probabilities[m][n];
        }
        codeword_decoded[n] = (L[n] < 0) ? 1 : 0;
    }
}

// Function to decode the message
void decode(pchk H, int *recv_codeword, int *codeword_decoded)
{
#ifdef TIMES
    clock_t clock_end;
    clock_t clock_start = clock();
#endif
    float *probabilities;
    float **M;
    float **extrinsic_probabilities;
    float *L;
    int try_n = 0;

    if (recv_codeword == NULL)
    {
#ifdef DEBUG
        printf("Not a valid codeword\n");
#endif
        return ;
    }
    if(check_codeword(H, recv_codeword) == 1){
        memcpy(codeword_decoded, recv_codeword, H.n_col * sizeof(int));
        return ;
    }

    // Initialize variables
    L = (float *)malloc(H.n_col * sizeof(float*));
    M = (float**)malloc(H.n_row * sizeof(float*));
    extrinsic_probabilities = (float**)malloc(H.n_row * sizeof(float*));
    for (int i = 0; i < H.n_row; i++)
    {
        extrinsic_probabilities[i] = (float*)malloc(H.n_col * sizeof(float));
        M[i] = (float*)malloc(H.n_col * sizeof(float));
    }

#ifdef TIMES
    clock_end = clock();
    //printf("memory initialization time:%ld\n",(clock_end-clock_start));
    printf(" %ld",(clock_end-clock_start));
    clock_start = clock();
#endif
    
#ifdef DEBUG
    printf("---------------------INITIALIZATIONS----------------------\n");
    


    printf("Codeword: ");
    print_vector_int(recv_codeword, H.n_col);
    printf("\n");
#endif

    //Compute LLR a priori probabilities (r)
    a_priori_probabilities(CURR_MODE, H.n_col , recv_codeword, &probabilities);

#ifdef DEBUG
    printf("Probabilities: ");
    print_vector_float(probabilities, H.n_col);
#endif

    // Initialize matrix M
    Initial_state( H, M, probabilities);

#ifdef TIMES
    clock_end = clock();
    //printf("initialization time:%ld\n",(clock_end-clock_start));
    printf(" %ld",(clock_end-clock_start));
    clock_start = clock();
#endif

#ifdef DEBUG
    printf("Matrix M: \n");
    print_matrix_float(M, H.n_row, H.n_col);
#endif



    while (try_n < MAX_ITERATIONS){
        try_n++;

#ifdef DEBUG
        printf("---------------------ITERATION %d----------------------\n", try_n);
#endif

        // Compute extrinsic probabilities matrix (E)
        compute_extrinsic(H,extrinsic_probabilities,M);
#ifdef DEBUG
        printf("Extrinsic probabilities: \n");
        print_matrix_float(extrinsic_probabilities, H.n_row, H.n_col);
#endif
        // Test
        Get_state(H,L,codeword_decoded,probabilities,extrinsic_probabilities);
#ifdef DEBUG
        printf("Final L: ");
        print_vector_float(L, H.n_col);
#endif
        // Check if it is a valid codeword
        if (check_codeword(H, codeword_decoded) == 1)
        {
#ifdef RESULT
            printf("Completed after %d iterations\n", try_n);
#endif
            break;
        }

        // Update matrix M
        Update_M(H,M,L,extrinsic_probabilities);
    }

#ifdef RESULT
    if(try_n == MAX_ITERATIONS)
        printf("Not completed after %d iterations\n", try_n);  
#endif
    codeword_decoded = NULL;

    free(probabilities);
    for (int i = 0; i < H.n_row; i++)
    {
        free(M[i]);
        free(extrinsic_probabilities[i]);
    }
    free(M);
    free(extrinsic_probabilities);

#ifdef TIMES
    clock_end = clock();
    //printf("x iterations time:%ld\n",(clock_end-clock_start));
    printf(" %ld",(clock_end-clock_start));
#endif

    return ;
}