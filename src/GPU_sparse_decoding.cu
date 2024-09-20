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
/*
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
        matrix->A[0] = (int *)malloc(matrix->type      *sizeof(int ));
        matrix->A[1] = (int *)malloc((matrix->n_row+1 )*sizeof(int ));

        
        fread(matrix->A[0],sizeof(int),matrix->type,f);
        fread(matrix->A[1],sizeof(int),matrix->n_row+1,f);
    }
    fclose(f);
    return NULL;
}

void print_vector_int(int vector[], int len)
{
    printf("[ ");
    for (int i = 0; i < len; i++)
        printf("%d ", vector[i]);
    printf("]\n");
}

 int main(int argc, char *argv[]){
    //check input arguments
    if(argc!=3){
        printf("Incorrect usage!\n Correct usage is: ./ldpc G_filepath H_filepath\n");
        exit(1);
    }

    //get parity check matrices from file
    pchk H,G;
    get_matrix_from_file(&G,argv[1]);
    get_matrix_from_file(&H,argv[2]);
    
    printf("filename:");
    printf(argv[2]);
    printf("\n");

    int *codeword_encoded = (int*)calloc(G.n_col,sizeof(int));

    int *codeword_decoded = (int*)calloc(G.n_col,sizeof(int));

    codeword_encoded[0] = 1 ;

    GPU_decode(H, codeword_encoded, codeword_decoded);
 }
 */