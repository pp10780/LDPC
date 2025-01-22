#include <stdio.h>
#include <time.h>

#include "GPU_decoding.h"
#include "defs.h"

//kernel 0: innit -> compute r and Li from m
//this is the same as cpu implementation
__global__ void GPU_sparse_apriori_probabilities(int n_col, float llr_i , int *m, float *r, float *L){
    //llr_i corresponds to the initial llr that's attributed depending on the channel (-llr_i if == 1) 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index > n_col)
        return;

    float r_val=(m[index]==0) ? llr_i : -llr_i;

    //write to global memory
    r[index] = r_val;
    L[index] = r_val;
}


//kernel 1: row wise -> compute M and "LE" from L and E, then compute E from M and "LE"
__global__ void GPU_sparse_row_wise(int n_row, int n_col, int *H, int *Hi, float *M, float* E, float *L, int *z, int *d_check){

    float LE = 1; //row value used to compute E
    int j = blockIdx.x * blockDim.x + threadIdx.x; //the thread's assigned row
    int check=0;
    float M_val,p;
    
    if(j > n_row)
        return;

    //do full row for M [first recursion]
    for (int i=Hi[j];i<Hi[j+1];i++){

        //early termination check (this is being done in parallel)
        check ^= z[H[i]];

        M_val = L[H[i]] - E[i];

        //store row value
        LE *= tanh(M_val/2);
        //writing result in global memory
        M[i] = M_val;
    }

    //do full row for E [second recursion]
    for (int i=Hi[j];i<Hi[j+1];i++){
        //exclude corresponding element from row
        p = LE/(tanh(M[i]/2) );
        E[i] = log((1+p)/(1-p));
    }

    //this is probably very bad maybe do a reduction?
    if(check == 1)
        *d_check=0;
}

//kernel 2: column wise -> compute L and z from E and r
__global__ void GPU_sparse_column_wise(int n_elements, int n_col, int *H, float* E, float* r,float *L, int *z){
    int i = (blockIdx.x * blockDim.x + threadIdx.x);//the thread's assigned column
    float L_val;//only write to global memory in the end

    if(i > n_col)
        return;
    
    L_val=r[i];
    //going column wise means going through the whole matrix H and if the index is the corresponding column then the element is part of the column
    for(int si=0; si<n_elements ; si++){
        if(H[si]==i)
            L_val+=E[si];
    }

    L[i] = L_val;
    z[i] = (L_val < 0) ? 1 : 0;;
}

/*
//kernel 1 without storing M
//kernel 1: row wise -> compute M and "LE" from L and E, then compute E from M and "LE"
__global__ void GPU_sparse_row_wise(int n_row, int n_col, int *H, int *Hi, float* E, float *L, int *z, int *d_check){

    float LE = 1; //row value used to compute E
    int j = blockIdx.x * blockDim.x + threadIdx.x; //the thread's assigned row
    int check=0;
    float p;
    
    if(j > n_row)
        return;

    //do full row for M [first recursion]
    for (int i=Hi[j];i<Hi[j+1];i++){

        //early termination check (this is being done in parallel)
        check ^= z[H[i]];
        
        //store row value
        LE *= tanh(L[H[i]] - E[i]/2);
    }

    //do full row for E [second recursion]
    for (int i=Hi[j];i<Hi[j+1];i++){
        //exclude corresponding element from row
        //p = LE/(tanh(M[i]/2) );
        p = LE/(tanh((L[H[i]] - E[i])/2) );
        E[i] = log((1+p)/(1-p));
    }

    //this is probably very bad maybe do a reduction?
    if(check == 1)
        *d_check=0;
}

//kernel 2: column wise using shared memory -> compute L and z from E and r
__global__ void GPU_sparse_column_wise(int n_elements, int n_col, int *H, float* E, float* r,float *L, int *z){
    int i = (blockIdx.x * blockDim.x + threadIdx.x);//the thread's assigned column
    int block_start = blockIdx.x * blockDim.x; //first column of the block (column for thread 0)

    //this stopped compiling with blockDim.x so I replaced it by 32 which is what it's going to be
    //__shared__ float b_L_val[blockDim.x]; //place where he L_cal is stored (each slot represents 1 column)
    //float t_L_val[blockDim.x];
    __shared__ float b_L_val[32];
    float t_L_val[32];

    //initiate thread memory
    for(int t=0;t<blockDim.x;t++)
        t_L_val[threadIdx.x]=0;

    //initiate shared memory
    if(i < n_col)
        b_L_val[threadIdx.x]=r[i];
    __syncthreads();

    //the whole matrix is split into each thread of the block
    //this needs to be rounded up so every element is present (later it will be verified if it goes over)
    int elements_per_thread = (n_elements+blockDim.x-1)/blockDim.x;
    
    //go through E column-wise only 1 block
    for(int si=elements_per_thread*threadIdx.x ; si<n_elements && si<elements_per_thread*(threadIdx.x+1); si++){
        if( 0 < H[si]-block_start  || H[si]-block_start < blockDim.x )
            t_L_val[H[si]-block_start]+=E[si];
    }

    //go through the shared memory in a round robin to get the full value of L
    int current;
    for(int t=0;t<blockDim.x;t++){
        current=threadIdx.x+t;
        if(current<blockDim.x)
            current-=blockDim.x;
        b_L_val[current]+=t_L_val[current];
        __syncthreads();
    }

    //up until this point "extra" threads were being used for the shared memory so they will now be purged
    if(i > n_col)
        return;
        
    L[i] = b_L_val[threadIdx.x];
    z[i] = (b_L_val[threadIdx.x] < 0) ? 1 : 0;;
}
*/

// Function to decode the message
void GPU_sparse_decode(pchk H, int *recv_codeword, int *codeword_decoded, float error_rate){
#ifdef TIMES
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif

#ifdef DEBUG
    float *matrix_debug_print=(float *)malloc(H.n_elements*sizeof(float));
    float *vector_debug_print=(float *)malloc(H.n_col*sizeof(float));
#endif
    //initialize device memory
    int check;
    
    //TODO: fix CL_NV_DEVICE_WARP_SIZE not working
    int threads_per_block= 32;//CL_NV_DEVICE_WARP_SIZE (not working for some reason)
    int rw_blocks=(H.n_col +threads_per_block -1)/threads_per_block;
    int cw_blocks=(H.n_row +threads_per_block -1)/threads_per_block;

    float init_prob=log((1 - error_rate)/error_rate);

    //decoding matrix
    int *dH;//indexes A[0]
    int *dHi;//row start and end (A[1])
    cudaMalloc((void **)&dH , H.n_elements* sizeof(int));
    cudaMalloc((void **)&dHi, (H.n_row+1) * sizeof(int));

    //computation matrices
    float *M,*E;
    //float *E;
    cudaMalloc((void **)&M, H.n_elements * sizeof(float));
    cudaMalloc((void **)&E, H.n_elements * sizeof(float));
    //E needs to be set at 0 at the start
    //TODO:this is not working!
    //cudaMemset(E,0,H.n_elements  * sizeof(int));

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
    cudaMemcpy( m   , recv_codeword , H.n_col       * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( dH  , H.A[0]        , H.n_elements  * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( dHi , H.A[1]        , (H.n_row+1)   * sizeof(int), cudaMemcpyHostToDevice);

#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(" memory initialization time:%3.3f \n",time*1000);
    //printf(" %ld",(clock_end-clock_start));
    cudaEventRecord(start, 0);
#endif

    //kernel 0:
    GPU_sparse_apriori_probabilities<<<rw_blocks, threads_per_block>>>(H.n_col, init_prob, m, r, L);
    cudaDeviceSynchronize();

#ifdef DEBUG
        printf("initialization:\n");
        printf("recv_codeword:[");
        for(int i=0;i<H.n_col;i++){
            printf("%d,",recv_codeword[i]);
        }
        printf("]\n\n");

        printf("m:[");
        cudaMemcpy(codeword_decoded,m,H.n_row*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i=0;i<H.n_col;i++){
            printf("%d,",codeword_decoded[i]);
        }
        printf("]\n\n");

        cudaMemcpy(vector_debug_print,L,H.n_row*sizeof(float),cudaMemcpyDeviceToHost);
        printf("L:[");
        for(int i=0;i<H.n_col;i++){
            printf("%f,",vector_debug_print[i]);
        }
        printf("]\n\n");

        cudaMemcpy(vector_debug_print,r,H.n_row*sizeof(float),cudaMemcpyDeviceToHost);
        printf("r:[");
        for(int i=0;i<H.n_col;i++){
            printf("%f,",vector_debug_print[i]);
        }
        printf("]\n\n");
#endif
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
    for (try_n = 0; try_n<MAX_ITERATIONS; try_n++){
        //set early termination do occur
        cudaMemset(d_check,1,sizeof(int));
        //kernel 1:                                        
        GPU_sparse_row_wise<<<rw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, dHi, M, E, L, z, d_check);
        //GPU_sparse_row_wise<<<rw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, dHi, E, L, z, d_check);

#ifdef DEBUG
        printf("iteration nº%d\n",try_n);
        cudaMemcpy(matrix_debug_print,M,H.n_elements*sizeof(float),cudaMemcpyDeviceToHost);
        //this is a print vector
        printf("M:[");
        for(int i=0;i<H.n_elements;i++){
            printf("%f,",matrix_debug_print[i]);
        }
        printf("]\n\n");

        cudaMemcpy(matrix_debug_print,E,H.n_elements*sizeof(float),cudaMemcpyDeviceToHost);
        //this is a print vector
        printf("E:[");
        for(int i=0;i<H.n_elements;i++){
            printf("%f,",matrix_debug_print[i]);
        }
        printf("]\n\n");
#endif
        cudaDeviceSynchronize();

        //kernel 2:
        GPU_sparse_column_wise<<<cw_blocks, threads_per_block>>>(H.n_elements, H.n_col, dH, E, r, L, z);

        //early termination (computing is done on kernel 1)
        cudaMemcpy(&check,d_check,1*sizeof(int),cudaMemcpyDeviceToHost);
        //running decoder for a set ammount of iterations
        if (check==1 && try_n!=0){
            //break;
            printf("solution was found!\n");
        }
            

        cudaDeviceSynchronize();
        
    }

    //get results from the device
    cudaMemcpy(codeword_decoded,z,H.n_row*sizeof(int),cudaMemcpyDeviceToHost);

#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf(" %d iterations time:%3.3f ",try_n,time*1000);
    //printf(" %ld",(clock_end-clock_start));
    cudaEventRecord(start, 0);
#endif

    return ;
}

//REMOVE : FOR TESTING PURPOSES ONLY!
//REMOVE : FOR TESTING PURPOSES ONLY!

int *generate_random_key(int size){
    int *key=(int *)malloc(size*sizeof(int));
    int i;

    for(i=0;i<size;i++)
        key[i] = rand()%2;

    return key;
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

void **get_matrix_from_file(pchk *matrix,char *filename){
    FILE *f = fopen (filename,"r");

    //Open file to read
    if(f==NULL){
        printf("couldn't open matrix file %s\n",filename);
        exit(1);
    }    

    //matrix info
    fread(&(matrix->n_row),sizeof(int),1,f);
    fread(&(matrix->n_col),sizeof(int),1,f);
    fread(&(matrix->n_elements),sizeof(int),1,f);
    fread(&(matrix->type),sizeof(int),1,f);

    if(matrix->type ==0){
        //normal
        matrix->A = (int**)malloc(matrix->n_row*sizeof(int*));
        for(int r=0;r<matrix->n_row;r++){
            matrix->A[r] = (int*)malloc(matrix->n_col*sizeof(int));
            fread(matrix->A[r],sizeof(int),matrix->n_col,f);
        }
    }
    else{
        //sparse
        matrix->A    = (int**)malloc(2                 *sizeof(int*));
        matrix->A[0] = (int *)malloc(matrix->n_elements*sizeof(int ));
        matrix->A[1] = (int *)malloc((matrix->n_row+1 ) *sizeof(int ));

        fread(matrix->A[0],sizeof(int),matrix->n_elements,f);
        fread(matrix->A[1],sizeof(int),matrix->n_row+1,f);
    }
    fclose(f);
    return NULL;
}

void dense_free_pchk(pchk mat){
    for(int i=0;i<mat.n_row;i++)
            free(mat.A[i]);
    free(mat.A);
}

void sparse_free_pchk(pchk mat){
    free(mat.A[0]);
    free(mat.A[1]);
    free(mat.A);
}


void free_pchk(pchk mat){
    switch(mat.n_elements){
        case 0://dense
            dense_free_pchk(mat);
            break;
        default://sparse
            sparse_free_pchk(mat);
    }
}

// Function to encode the key
void encode(int *key, pchk generator, int *message)
{
    if(generator.type == 0){
        //normal
        for(int row = 0; row < generator.n_row; row++){
            for(int col = 0; col < generator.n_col; col++){
                if(generator.A[row][col]==1)
                    message[row] ^= key[col];
            }
        }
    }
    else{
        //sparse
        for(int r=0;r<generator.n_row;r++){
            for (int c = generator.A[1][r]; c <  generator.A[1][r+1]; c++)
                message[r] ^= key[ generator.A[0][c] ];
        }
            
    }

}

int main(int argc, char *argv[]){
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
    int *key = generate_random_key(key_size);
    
    int *codeword_encoded   = (int*)calloc(message_size,sizeof(int));
    int *codeword_decoded   = (int*)calloc(message_size,sizeof(int));
    int *transmitted_mesage;


    //ENCDODING
    if(g_flag)
        encode((int *)key, G, codeword_encoded);

    //TRANSMISSIONs
    transmitted_mesage = add_error(codeword_encoded,message_size,error_rate,max_errors);
        
    //DECODING
#ifdef TIMES
    struct timespec clock_begin, clock_end;
    clock_gettime(CLOCK_REALTIME, &clock_begin);
#endif

    GPU_sparse_decode(H, transmitted_mesage, codeword_decoded,error_rate);

    
#ifdef TIMES
    clock_gettime(CLOCK_REALTIME, &clock_end);
    long seconds = clock_end.tv_sec - clock_begin.tv_sec;
    long nanoseconds = clock_end.tv_nsec - clock_begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;
    printf("decoding time: %f\n",elapsed);
#endif

    if(codeword_decoded == NULL){
        printf("Not a valid codeword\n");
        return 0;
    }

    //check result
    int correct=1;
    int c;
    for(c=0;c<message_size;c++){
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
    //free(codeword_decoded);

    if(correct)
        return 0;
    return 0;
}