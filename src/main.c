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

int* add_error(int *codeword,int codeword_size,float error_rate,int max_errors){
    int inverse=(1/error_rate),counter=0;
    int *transmitted_mesage = (int*)malloc(codeword_size * sizeof(int));;

    for(int c=0;c<codeword_size;c++){
        //error
        if(rand() % inverse == 0 && (counter < max_errors || max_errors== -1) ){
            transmitted_mesage[c] = !codeword[c];
            counter++;
        }
        else
            transmitted_mesage[c] = codeword[c];
    }

    printf("added %d errors\n",counter);        
    return transmitted_mesage;
}

int main(int argc, char *argv[])
{
    float error_rate= DEFAULT_ERROR_RATE;
    int max_errors = DEFAULT_MAX_ERRORS;
    int g_flag=1;
    int key_size=0,message_size=0;
    //check input arguments
    if(argc<3 || argc>5){
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

    key_size=G.n_row;
    message_size=G.n_col;

    if(G.n_col != H.n_col){
        printf("coding and decoding matrices do not match!\n using a '0's message\n");
        message_size=H.n_col;
        key_size=H.n_row;
        g_flag=0;
    }

#ifdef DEBUG
    if(g_flag){
        printf("G:\n");
        print_parity_check(G);
    }

    printf("\n");
    printf("H:\n");
    print_parity_check(H);
    printf("\n");

#endif

    srand(time(NULL));
    int *message = generate_random_key(key_size);
#ifdef DEBUG
    printf("message to be encoded:\n");
    print_vector_int(message,key_size);
#endif
    
    int *codeword_encoded   = (int*)calloc(message_size,sizeof(int));
    int *codeword_decoded   = (int*)calloc(message_size,sizeof(int));
    int *transmitted_mesage;


    //ENCDODING
    if(g_flag)
        encode((int *)message, G, codeword_encoded);

#ifdef RESULT
    print_vector_int(codeword_encoded, message_size);
#endif

    //TRANSMISSIONs
    transmitted_mesage = add_error(codeword_encoded,message_size,error_rate,max_errors);


        
#ifdef RESULT
    print_vector_int(transmitted_mesage, message_size);
#endif 
    //DECODING
#ifdef TIMES
    clock_t clock_start = clock();
#endif
    if(H.type == 0){
#ifndef GPU
        decode(H, transmitted_mesage, codeword_decoded,error_rate);
#endif
#ifdef GPU
        GPU_decode(H, transmitted_mesage, codeword_decoded);
#endif
    }
    else{
        sparse_decode(H,transmitted_mesage,codeword_decoded,error_rate);
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
    print_vector_int(codeword_decoded, message_size);
#endif

    //check result
    int correct=1;
    for(int c=0;c<message_size;c++){
        if(codeword_encoded[c] != codeword_decoded[c]){
            printf("decoding is incorrect!\n");
            correct=0;
            break;
        }
    }
    if(correct)
        printf("decoding is correct!\n");

    //TODO: have actual big G matrices
    free_pchk(G);
    free_pchk(H);

    free(message);
    free(codeword_encoded);
    free(codeword_decoded);

    if(correct)
        return 1;
    return 0;
}