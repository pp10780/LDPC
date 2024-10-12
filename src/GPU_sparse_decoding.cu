#include <stdio.h>
#include <time.h>


extern "C" {
#include "GPU_decoding.h"
#include "defs.h"
}

//kernel 0: innit -> compute r and Li from m
//this is the same as sparse implementation
__global__ void GPU_sparse_apriori_probabilities(int n_col, float llr_i , int *m, float *r, float *L){
    //llr_i corresponds to the initial llr that's attributed depending on the channel (-llr_i if == 1) 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index > n_col)
        return;
    //TODO:this could be fancier by just changing the signal bit according to the data bit 
    float r_val=(m[index]==0) ? llr_i : -llr_i;

    //write to global memory
    r[index] = r_val;
    L[index] = r_val;
}


//kernel 1: row wise -> compute M and "LE" from L and E, then compute E from M and "LE"
__global__ void GPU_sparse_row_wise(int n_row, int n_col, int *H, int *Hi, float *M, float* E, float *L, int *z){

    float LE = 1; //row value used to compute E
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = n_col*j;
    int check=0;
    
    if(j > n_row)
        return;

    //do full row for M [first recursion]
    for (int i=Hi[j];i<Hi[j+1];i++){

        //early termination check (this is being done in parallel)
        check ^= z[H[i]];

        float M_val = L[H[i]] - E[i];

        //store row value
        LE *= tanh(M_val/2);
        //writing result in global memory
        M[i] = M_val;
    }

    //do full row for E [second recursion]
    for (int i=Hi[j];i<Hi[j+1];i++){
        //exclude corresponding element from row
        float p = LE/(tanh(M[i]/2) );
        E[i] = log((1+p)/(1-p));
    }

    //this is probably very bad maybe do a reduction?
    if(check == 1)
        *d_check=0;
}

//kernel 2: column wise -> compute L and z from E and r
__global__ void GPU_sparse_column_wise(int n_elements, int *H, float* E, float* r,float *L, int *z){
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    float L_val;//only write to global memory in the end

    if(i > n_col)
        return;
    
    L_val=r[i];
    //go through E column wise -> this is terrebly inefficient in csr can't find anyone doing it different still in csr
    //going column wise means going through the whole matrix H and if the index the corresponding column it is part of the column
    for(int si=0; si<n_elements ; si++){
        if(H[si]==i)
            L_val+=E[i];
    }

    L[i] = L_val;
    z[i] = (L_val < 0) ? 1 : 0;;
}

// Function to decode the message
extern "C"
void GPU_sparse_decode(pchk H, int *recv_codeword, int *codeword_decoded){
#ifdef TIMES
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    //initialize device memory
    int check;
    
    //TODO: this is temporary I still need to calculate the number of blocks required and set the number of threads per block in defs
    int threads_per_block=32;//CL_NV_DEVICE_WARP_SIZE (not working for some reason)
    //TODO: make this better
    int rw_blocks=H.n_col/threads_per_block + 1;
    int cw_blocks=H.n_row/threads_per_block + 1;
    int MAX_ITERATION = 10;

    float init_prob=log((1 - BSC_ERROR_RATE)/BSC_ERROR_RATE);

    //decoding matrix
    int *dH;//indexes A[0]
    int *dHi;//row start and end (A[1])
    cudaMalloc((void **)&dH , H.type      * sizeof(int));
    cudaMalloc((void **)&dHi, (H.n_row+1) * sizeof(int));

    //computation matrices
    float *M,*E;
    cudaMalloc((void **)&M, H.type * sizeof(float));
    cudaMalloc((void **)&E, H.type * sizeof(float));
    //E needs to be set at 0 at the start
    cudaMemset(M,0,H.type  * sizeof(int));

    //vectors
    float *r,*L;
    int   *z,*m;
    cudaMalloc((void **)&r, H.n_col * sizeof(float));
    cudaMalloc((void **)&L, H.n_col * sizeof(float));
    cudaMalloc((void **)&z, H.n_col * sizeof(int));
    cudaMalloc((void **)&m, H.n_col * sizeof(int));

    //check
    int *d_check;
    cudaMalloc((void **)&d_check, 1* sizeof(int));
    cudaMemset(d_check,1,sizeof(int));

    //load inital data to device
    cudaMemcpy(m, recv_codeword, H.n_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( dH , H.A[0], H.type      * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( dHi, H.A[1], (H.n_row+1) * sizeof(int), cudaMemcpyHostToDevice);

#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(" memory initialization time:%3.3f\n",time*1000);
    //printf(" %ld",(clock_end-clock_start));
    cudaEventRecord(start, 0);
#endif

    //kernel 0:
    GPU_apriori_probabilities<<<rw_blocks, threads_per_block>>>(H.n_col, init_prob, m, r, L);
    cudaDeviceSynchronize();

#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(" initialization time:%3.3f\n",time*1000);
    //printf(" %ld",(clock_end-clock_start));
    cudaEventRecord(start, 0);
#endif

    //iterative portion
    int try_n;
    for (try_n = 0; try_n<MAX_ITERATION; try_n++){
        //set early termination do occur
        cudaMemset(d_check,1,sizeof(int));
        //kernel 1:
        GPU_sparse_row_wise<<<rw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, dHi, M, E, L, z);
        cudaDeviceSynchronize();

        //kernel 2:
        GPU_sparse_column_wise<<<cw_blocks, threads_per_block>>>(H.type, dH, E, r, L, z);
        
        if (check==1 && try_n!=0)
            break;
        
        cudaDeviceSynchronize();

        //kernel 3: -> this was merged into kernel 1
        //early_termination<<<rw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, z, d_check);
        //cudaDeviceSynchronize();
        //cudaMemcpy(&check,d_check,1*sizeof(int),cudaMemcpyDeviceToHost);
        
    }

    //get results from the device
    cudaMemcpy(codeword_decoded,z,H.n_row*sizeof(int),cudaMemcpyDeviceToHost);

#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(" %d iterations time:%3.3f\n",try_n,time*1000);
    //printf(" %ld",(clock_end-clock_start));
    cudaEventRecord(start, 0);
#endif

    return ;
}


//REMOVE : FOR TESTING PURPOSES ONLY!

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

    key_size=G.n_col;
    message_size=G.n_row;

    if(G.n_row != H.n_col){
           message_size=H.n_col;
        key_size=H.n_row;
        g_flag=0;
        printf("coding and decoding matrices do not match!\nusing a '0's message with size:%d\n",message_size);

    }

    srand(time(NULL));
    //int *key = generate_random_key(key_size);
    int *key=(int *)calloc(size,sizeof(int));
    
    int *codeword_encoded   = (int*)calloc(message_size,sizeof(int));
    int *codeword_decoded   = (int*)calloc(message_size,sizeof(int));
    int *transmitted_mesage;

    GPU_decode(H, transmitted_mesage, codeword_decoded);
    //ENCDODING
    //if(g_flag)
    //    encode((int *)key, G, codeword_encoded);

    //TRANSMISSIONs
    transmitted_mesage = add_error(codeword_encoded,message_size,error_rate,max_errors);
    
#ifdef TIMES
    clock_t clock_end = clock();
    //printf("decoding time: %ld\n",(clock_end-clock_start));
    printf(" %ld\n",(clock_end-clock_start));

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

    free_pchk(G);
    free_pchk(H);

    free(key);
    free(codeword_encoded);
    free(codeword_decoded);

    if(correct)
        return 1;
    return 0;
}