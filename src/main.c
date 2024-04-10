#include <time.h>
#include "encoding.h"
#include "decoding.h"
#include "display_variables.h"
#include "defs.h"
#include "storage.h"

#include "sparse_decoding.h"

//this is to go in the seperate file
#include <string.h>

void BSC_noise(int *codeword, float p)
{
    srand(time(NULL));
    for (int i = 0; i < CODEWORD_LEN; i++)
    {
        float r = (float)rand() / RAND_MAX;
        printf("r = %f\n", r);
        if (r < p)
        {
            codeword[i] = !codeword[i];
        }
    }
}

void *Transpose_pchk(pchk *Tmat,pchk mat){
    int *row_count;

    //fill out matrix info
    Tmat->n_row = mat.n_col;
    Tmat->n_col = mat.n_row;
    Tmat->type  = mat.type;

    //allocate space
    Tmat->A    = (int **)malloc(2             *sizeof(int*));
    Tmat->A[0] = (int * )malloc(Tmat->type    *sizeof(int ));
    Tmat->A[1] = (int * )malloc(Tmat->n_row+1 *sizeof(int ));
    row_count  = (int * )calloc(Tmat->n_row   ,sizeof(int ));
    

    //count number of elements per row of mat
    for(int e=0;e<mat.type;e++)
        row_count[mat.A[0][e]]++;

    //convert number of elements into row indexes
    Tmat->A[1][0]=0;
    for(int r=1;r <= Tmat->n_row;r++)
        Tmat->A[1][r]  = row_count[r-1] + Tmat->A[1][r-1];

    //fill out elements in their apropriate spots
    for(int r=0; r<mat.n_row; r++){
        for(int c=mat.A[1][r]; c<mat.A[1][r+1]; c++){
            int tr = mat.A[0][c];
            int index = Tmat->A[1][tr+1] - row_count[tr];
            Tmat->A[0][ index ] = r;
            row_count[tr] --;
        }
    }
    
    //for some reason I can't seem to free this vector
    //free(row_count);
    return NULL;
}

int main(int argc, char *argv[])
{
    //check input arguments
    if(argc!=3){
        printf("Incorrect usage!\n Correct usage is: ./ldpc G_filepath H_filepath\n");
        exit(1);
    }

    //get parity check matrices from file
    pchk H,G,TH;
    get_matrix_from_file(&G,argv[1]);
    get_matrix_from_file(&H,argv[2]);

#ifdef DEBUG
    print_parity_check(G);
    printf("\n");
    print_parity_check(H);
    printf("\n");
#endif
    int message[MESSAGE_LEN] = {0, 0, 0};
    
    int *codeword_encoded = (int*)calloc(CODEWORD_LEN,sizeof(int));

    int *codeword_decoded = (int*)calloc(CODEWORD_LEN,sizeof(int));

    //int test_codeword[CODEWORD_LEN] = {1, 1, 1, 1, 1, 1};

    encode((int *)message, G, codeword_encoded);
    
    print_vector_int(codeword_encoded, CODEWORD_LEN);

    //memcpy(codeword_encoded, test_codeword, CODEWORD_LEN*sizeof(int));

    BSC_noise(codeword_encoded, BSC_ERROR_RATE);

    print_vector_int(codeword_encoded, CODEWORD_LEN);
    if(H.type == 0){
        decode(H, codeword_encoded, codeword_decoded);
    }
    else{
        printf("there\n");
        Transpose_pchk(&TH,H);
        printf("here\n");
#ifdef DEBUG
printf("\n");
        print_parity_check(H);
        printf("\n");
        print_parity_check(TH);
        printf("\n");
#endif
        sdecode(H,TH,codeword_encoded,codeword_decoded);
    }
        

    if(codeword_decoded == NULL)
    {
        printf("Not a valid codeword\n");
        return 0;
    }
    print_vector_int(codeword_decoded, CODEWORD_LEN);

    free(codeword_encoded);
    free(codeword_decoded);

    //check_possible_codewords(H);
    return 0;
}