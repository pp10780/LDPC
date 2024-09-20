#include <time.h>
#include "encoding.h"
#include "decoding.h"
#include "display_variables.h"
#include "defs.h"
#include "storage.h"

#include "sparse_decoding.h"

#ifdef GPU
#include "GPU_decoding.cu"
#endif

int *generate_random_key(int size){
    int *key=(int *)malloc(size*sizeof(int));

    for(int i=0;i<size;i++)
        key[i] = rand()%2;

    return key;
}

/*
this was for adding a specific number of errors to the message
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
            pos[i]=new_pos;
            i++;
            codeword[new_pos]=!codeword[new_pos];
        }
    }
    free(pos);
    return;
}
*/

void add_error(int *codeword,int codeword_size,float error_rate,int max_errors){
    int inverse=(1/error_rate),counter=0;
    for(int c=0;c<codeword_size;c++){
        //error
        if(rand() % inverse == 0){
            codeword[c] = !codeword[c];
            counter++;
            if(counter >= max_errors && max_errors!= -1)
                break;
        }
        
    }
    printf("added %d errors\n",counter);        
    return;
}

int main(int argc, char *argv[])
{
    float error_rate= DEFAULT_ERROR_RATE;
    int max_errors = DEFAULT_MAX_ERRORS;
    //check input arguments
    if(argc<3 && argc>5){
        printf("Incorrect usage!\n Correct usage is: ./ldpc G_filepath H_filepath [error rate] [max errors]\n");
        exit(1);
    }
    if(argc>3)
        error_rate=atof(argv[3]);
    if(argc>4)
        max_errors=atoi(argv[4]);

    //get parity check matrices from file
    pchk H,G;
    get_matrix_from_file(&G,argv[1]);
    get_matrix_from_file(&H,argv[2]);

#ifdef DEBUG
    //TODO: have a actual big matrix G
    //printf("G:\n");
    //print_parity_check(G);

    printf("\n");
    printf("H:\n");
    print_parity_check(H);
    printf("\n");

#endif

    //TODO:have actual big matrix G
    G.n_col=H.n_col;
    G.n_row=H.n_row;

    srand(time(NULL));
    int *message = generate_random_key(G.n_row);
#ifdef DEBUG
    printf("message to be encoded:\n");
    print_vector_int(message,G.n_row);
#endif
    
    int *codeword_encoded = (int*)calloc(G.n_col,sizeof(int));
    int *codeword_decoded = (int*)calloc(G.n_col,sizeof(int));


    //ENCDODING
    //TODO: have actual big matrix G
    encode((int *)message, G, codeword_encoded);

#ifdef RESULT
    print_vector_int(codeword_encoded, G.n_col);
#endif

    //TRANSMISSIONs
    add_error(codeword_encoded,G.n_col,error_rate,max_errors);


        
#ifdef RESULT
    print_vector_int(codeword_encoded, G.n_col);
#endif 
    //DECODING
#ifdef TIMES
    clock_t clock_start = clock();
#endif
    if(H.type == 0){
#ifndef GPU
        decode(H, codeword_encoded, codeword_decoded,error_rate);
#endif
#ifdef GPU
        GPU_decode(H, codeword_encoded, codeword_decoded);
#endif
    }
    else{
        sparse_decode(H,codeword_encoded,codeword_decoded,error_rate);
    }

#ifdef TIMES
    clock_t clock_end = clock();
    //printf("decoding time: %ld\n",(clock_end-clock_start));
    printf(" %ld\n",(clock_end-clock_start));

#endif

    if(codeword_decoded == NULL){
        printf("Not a valid codeword\n");
        return 0;
    }


#ifdef RESULT
    print_vector_int(codeword_decoded, G.n_col);
#endif
    //TODO: have actual big G matrices
    //free_pchk(G);
    free_pchk(H);

    free(message);
    free(codeword_encoded);
    free(codeword_decoded);

    //check_possible_codewords(H);
    return 0;
}