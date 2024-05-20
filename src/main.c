#include <time.h>
#include "decoding.h"
#include "display_variables.h"
#include "defs.h"
#include "storage.h"
#include "simple_decoding.h"
#include "simple_operations.h"
#include "sparse_decoding.h"

//this is to go in the seperate file
#include <string.h>

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
    pchk H,G;
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

    //encoding message
    //encode((int *)message, G, codeword_encoded);
    mod2_vectmatmul(message,G,codeword_encoded);
    print_vector_int(codeword_encoded, G.n_col);

    //transmiting message
    //add_error(codeword_encoded,G.n_col,0);
    int e[6]={1,0,0,0,0,0};
    bitwise_vectors(codeword_encoded,codeword_encoded,(int *)e, G.n_col);
    print_vector_int(codeword_encoded, G.n_col);

    //decoding message
    if(H.type == 0){
        simple_decode(H, codeword_encoded, codeword_decoded);
        //decode(H, codeword_encoded, codeword_decoded);

        //free matrices accordin to how they're done
        for(int i=0;i<H.n_row;i++)
            free(H.A[i]);
        free(H.A);

        for(int i=0;i<G.n_row;i++)
            free(G.A[i]);
        free(G.A);
    }
    else{
        sparse_decode(H,codeword_encoded,codeword_decoded);

        //free matrices accordin to how they're done
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
    printf("decoded message:\n");
    print_vector_int(codeword_decoded,  G.n_col);
    printf("original message:\n");
    print_vector_int(message,  G.n_row);

    free(message);
    free(codeword_encoded);
    free(codeword_decoded);

    return 0;
}