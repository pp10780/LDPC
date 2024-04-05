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
    int members,*buffer;

    Tmat->n_row = mat.n_col;
    Tmat->n_col = mat.n_row;
    Tmat->type = 1;
    Tmat->A = (int **) malloc(sizeof(int*) * Tmat->n_row);
    buffer = (int *) malloc(sizeof(int) * Tmat->n_col+1);

    for(int tr=0;tr<Tmat->n_row;tr++){
        members=0;
        for(int r=0;r<mat.n_row;r++){
            for(int c=1;c<mat.A[r][0]+1;c++){
                if(mat.A[r][c] == tr){
                    buffer[members] = r;
                    members++;
                    break;
                }
                if(mat.A[r][c] > tr)
                    break;
            }
        }
        Tmat->A[tr] = (int *) malloc(sizeof(int) * members+1);
        Tmat->A[tr][0] = members;
        for(int tc=0;tc < members;tc++)
            Tmat->A[tr][tc+1] = buffer[tc]; 
            
    }

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
    
    if(H.type==1){
        Transpose_pchk(&TH,H);
#ifdef DEBUG
        printf("\n");
        print_parity_check(TH);
        printf("\n");
#endif
        sdecode(H,TH,codeword_encoded,codeword_decoded);
    }
    else
        decode(H, codeword_encoded, codeword_decoded);

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