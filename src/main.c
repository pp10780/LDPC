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
    Tmat->A    = (int **)malloc(2              *sizeof(int*));
    Tmat->A[0] = (int * )malloc(Tmat->type     *sizeof(int ));
    Tmat->A[1] = (int * )malloc((Tmat->n_row+1)*sizeof(int ));
    row_count  = (int * )calloc(Tmat->n_row    ,sizeof(int ));
    

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

int *generate_random_key(int size){
    int *key=(int *)malloc(size*sizeof(int));

    for(int i=0;i<size;i++)
        key[i] = rand()%2;


    return key;
}

void add_error(int *codeword,int codeword_size,int n_errors){
    int *pos;
    int new_pos,flag;

    if(codeword_size < n_errors){
        printf("too much error is being added\n");
        return;
    }
    pos=(int *)malloc(n_errors*sizeof(int));

    for(int i=0;i<n_errors;){
        new_pos = rand() % codeword_size;
        flag=1;
        for(int c=0;c<i;c++){
            if(pos[c] == new_pos){
                flag=0;
                break;
            }
        }
        if(flag){
            i++;
            codeword[i]=!codeword[i];
        }
    }
    free(pos);
    return;
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
    printf("G:\n");
    print_parity_check(G);
    printf("\n");
    printf("H:\n");
    print_parity_check(H);
    printf("\n");
#endif
    srand(time(NULL));
    int *message = generate_random_key(G.n_row);
#ifdef DEBUG
    printf("message to be encoded:\n");
    print_vector_int(message,G.n_row);
#endif
    
    int *codeword_encoded = (int*)calloc(G.n_col,sizeof(int));

    int *codeword_decoded = (int*)calloc(G.n_col,sizeof(int));

    encode((int *)message, G, codeword_encoded);
    
    print_vector_int(codeword_encoded, G.n_col);

    add_error(codeword_encoded,G.n_col,1);

    print_vector_int(codeword_encoded, G.n_col);
    if(H.type == 0){
        decode(H, codeword_encoded, codeword_decoded);
    }
    else{
        Transpose_pchk(&TH,H);
#ifdef DEBUG
printf("\n");
        printf("Transposed H:\n");
        print_parity_check(TH);
        printf("\n");
#endif
        sparse_decode(H,codeword_encoded,codeword_decoded);
        free(TH.A[0]);
        free(TH.A[1]);
        free(TH.A);

        free(H.A[0]);
        free(H.A[1]);
        free(H.A);

        free(G.A[0]);
        free(G.A[1]);
        free(G.A);
    }
        

    if(codeword_decoded == NULL)
    {
        printf("Not a valid codeword\n");
        return 0;
    }
    print_vector_int(codeword_decoded, CODEWORD_LEN);

    free(message);
    free(codeword_encoded);
    free(codeword_decoded);

    //check_possible_codewords(H);
    return 0;
}