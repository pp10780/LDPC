#include <stdio.h>
#include <time.h>

#include "encoding.h"
#include "decoding.h"
#include "display_variables.h"
#include "defs.h"
#include "storage.h"

#include "sparse_decoding.h"

#ifdef GPU
#include "GPU_sparse_decoding.h"
#endif

int iterations;

int *generate_random_key(int size){
    int *key=(int *)malloc(size*sizeof(int));
    int i;

    for(i=0;i<size;i++)
        key[i] = rand()%2;

    return key;
}

int *generate_error_key(int size,float error_rate,int max_errors){
    int *error_key = (int*)calloc(size,sizeof(int));
    int random_pos;
    int num_errors;

    if(max_errors ==-1)
        max_errors=size*error_rate;

    for(num_errors=0; num_errors<max_errors; num_errors++){
        for( random_pos= rand()%size; error_key[random_pos]==1;random_pos++){
            if(random_pos+1 > size)
                random_pos=-1;
        }
        error_key[random_pos]=1;
    }
#ifdef VERBOSE
    printf("added %d errors\n",num_errors);  
#endif
    return error_key;
}

int* add_error(int *codeword,int codeword_size,float error_rate,int max_errors){
    int inverse=(1/error_rate),counter=0;
    int *transmitted_mesage = (int*)malloc(codeword_size * sizeof(int));

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

int main(int argc, char *argv[]){
    float error_rate= DEFAULT_ERROR_RATE;
    int max_errors = DEFAULT_MAX_ERRORS;
    int g_flag=1;
    int key_size=0,message_size=0;
    //check input arguments
    if(argc<3 || argc>6){
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

    key_size=G.n_col;
    message_size=G.n_row;

    if(G.n_row != H.n_col){
           message_size=H.n_col;
        key_size=H.n_row;
        g_flag=0;
        #ifdef VERBOSE
        printf("coding and decoding matrices do not match!\nusing a '0's message with size:%d\n",message_size);
        #endif
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

    //srand(time(NULL));
    srand(atoi(argv[5]));
    int *key = generate_random_key(key_size);
#ifdef RESULT
    printf("key to be encoded:\n");
    print_vector_int(key,key_size);
#endif
    
    int *codeword_encoded   = (int*)calloc(message_size,sizeof(int));
    int *codeword_decoded   = (int*)calloc(message_size,sizeof(int));
    int *transmitted_mesage = (int*)calloc(message_size,sizeof(int));
    int *error_key;


    //ENCDODING
    if(g_flag)
        encode((int *)key, G, codeword_encoded);

#ifdef RESULT
    printf("encoded message:\n");
    print_vector_int(codeword_encoded, message_size);
#endif

    //TRANSMISSION
    //error_key = add_error(codeword_encoded,message_size,error_rate,max_errors);
    error_key = generate_error_key(message_size,error_rate,max_errors);

    for(int i=0;i<message_size;i++)
        transmitted_mesage[i]=error_key[i]^codeword_encoded[i];
    
    
#ifdef RESULT
    printf("error key:\n");
    print_vector_int(error_key, message_size);
    printf("transmitted message:\n");
    print_vector_int(transmitted_mesage, message_size);
#endif 
    free(error_key);

    //this will be the return value to evaluate performance
    iterations=-1;
    //DECODING
    if(H.type == 1){
#ifndef GPU
        //int tester[6] = {0,0,0,0,1,0};
        //sparse_decode(H, tester, codeword_decoded,error_rate);
        iterations=sparse_decode(H, transmitted_mesage, codeword_decoded,error_rate);
#endif
#ifdef GPU
        // int tester[6] = {0,0,1,1,1,0};
        // GPU_sparse_decode(H, tester, codeword_decoded,&error_rate);
        iterations=GPU_sparse_decode(H, transmitted_mesage, codeword_decoded,&error_rate);
#endif
    }
    else{
        decode(H,transmitted_mesage,codeword_decoded,error_rate);
    }

    if(codeword_decoded == NULL){
        printf("Not a valid codeword\n");
        return 0;
    }


#ifdef RESULT
    print_vector_int(codeword_decoded, message_size);
#endif

    //check result
    int correct=1;
    int c;
    for(c=0;c<message_size;c++){
        if(codeword_encoded[c] != codeword_decoded[c]){
#ifdef VERBOSE
            printf("decoding is incorrect!\n");
#endif
            correct=0;
            break;
        }
    }
    if(correct)
#ifdef VERBOSE
        printf("decoding is correct!\n");
#endif

    free_pchk(G);
    free_pchk(H);

    free(key);
    free(codeword_encoded);
    free(transmitted_mesage);
    free(codeword_decoded);

    if(correct)
        return iterations;
    return -1;
}
