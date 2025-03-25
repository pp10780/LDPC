#include <stdio.h>
#include <time.h>

#include "GPU_sparse_decoding.h"
#include "defs.h"

//CL_NV_DEVICE_WARP_SIZE (not working for some reason)
//TODO: fix CL_NV_DEVICE_WARP_SIZE not working
const int THREADS_PER_BLOCK=32;

//kernel 0: innit -> compute r and Li from m
__global__ void GPU_sparse_apriori_probabilities(int n_col, float llr_i , int *m, float *r, float *L, int *z){
    //llr_i corresponds to the initial llr that's attributed depending on the channel (-llr_i if == 1) 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index > n_col)
        return;

    float r_val=(m[index]==0) ? llr_i : -llr_i;
    //write to global memory
    r[index] = r_val;
    L[index] = r_val;
    z[index]=m[index];
}

//kernel 1: row wise -> compute M and "LE" from L and E, then compute E from M and "LE"
__global__ void GPU_sparse_row_wise(int n_row, int n_col, int *H, int *Hi, float* E, float *L, int *z, int *d_check){
    float LE = 1; //row value used to compute E
    int j = blockIdx.x * blockDim.x + threadIdx.x; //the thread's assigned row
    int check=0;
    float p;

    if(j < n_row){
        //do full row for M [first recursion]
        for (int i=Hi[j];i<Hi[j+1];i++){

            //early termination check (this is being done in parallel)
            check ^= z[H[i]];
            //store row value
            LE *= tanh((L[H[i]] - E[i])/2);
        }

        //do full row for E [second recursion]
        for (int i=Hi[j];i<Hi[j+1];i++){
            //exclude corresponding element from row
            //p = LE/(tanh(M[i]/2) );
            p  = LE/(tanh((L[H[i]] - E[i])/2) );
            
            E[i] = log((1+p)/(1-p));
        }

        //TODO: use unified memory on this
        if(check == 1){
            *d_check=1;
        }
            
    }
    
}

/*
//without using shared memory
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
    z[i] = (L_val < 0) ? 1 : 0;
}
*/

//kernel 2: column wise using shared memory -> compute L and z from E and r
__global__ void GPU_sparse_column_wise(int n_elements, int n_col, int *H, float* E, float* r,float *L, int *z){
    int i = (blockIdx.x * blockDim.x + threadIdx.x);//the thread's assigned column
    int block_start = blockIdx.x * blockDim.x; //first column of the block (column for thread 0)

    __shared__ float b_L_val[THREADS_PER_BLOCK];
    float t_L_val[THREADS_PER_BLOCK];

    //initiate thread memory
    for(int t=0;t<blockDim.x;t++)
        t_L_val[t]=0;

    //initiate shared memory
    if(i < n_col)
        b_L_val[threadIdx.x]=r[i];
    __syncthreads();

    //the whole matrix is split into each thread of the block
    //this needs to be rounded up so every element is present (later it will be verified if it goes over)
    int elements_per_thread = (n_elements+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    
    //go through E column-wise only 1 block
    for(int si=elements_per_thread*threadIdx.x ; si<n_elements && si<elements_per_thread*(threadIdx.x+1); si++){
        //this is the index in relation to the warp  
        int si_id = H[si]-block_start;

        //check if this element belong to the warp and include it if so
        if( 0 <= si_id && si_id < THREADS_PER_BLOCK )
            t_L_val[si_id]+=E[si];
    }
    
    //go through the shared memory in a round robin to get the full value of L
    int current;
    for(int t=0;t<blockDim.x;t++){
        current=threadIdx.x+t;
        if(current >= THREADS_PER_BLOCK)
            current-=blockDim.x;
        b_L_val[current]+=t_L_val[current];
        __syncthreads();
    }
    
    //up until this point "extra" threads were being used for the shared memory so they will now be purged
    if(i > n_col)
        return;
        
    L[i] = b_L_val[threadIdx.x];
    z[i] = (b_L_val[threadIdx.x] < 0) ? 1 : 0;
}


// Function to decode the message
extern "C" int GPU_sparse_decode(pchk H, int *recv_codeword, int *codeword_decoded, float *error_rate){


#ifdef TIMES
    float time,tmememory,k0,k1=0,k2=0;
    cudaEvent_t start, stop, start2, stop2;
    //FILE *log;
    //log = fopen("times.txt", "a");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start, 0);
#endif

#ifdef DEBUG
    cudaError_t error1,error2;
    float   *matrix_debug_print=(float *)malloc(H.n_elements*sizeof(float));
    float   *vector_debug_print=(float *)malloc(H.n_col*sizeof(float));
    int     *index_debug_print=(int   *)malloc(H.n_elements*sizeof(int));
#endif

    float init_prob=log((1 - *error_rate)/ *error_rate);
    //thread control
    int threads_per_block= THREADS_PER_BLOCK;
    int rw_blocks=(H.n_col +threads_per_block -1)/threads_per_block;
    int cw_blocks=(H.n_col +threads_per_block -1)/threads_per_block;

    //initialize device memory

    //decoding matrix
    int *dH;//indexes A[0]
    int *dHi;//row start and end (A[1])
    cudaMalloc((void **)&dH , H.n_elements* sizeof(int));
    cudaMalloc((void **)&dHi, (H.n_row+1) * sizeof(int));

    //computation matrices
    float *E;
    cudaMalloc((void **)&E, H.n_elements * sizeof(float));
    //E needs to be set at 0 at the start
    cudaMemset(E,0,H.n_elements  * sizeof(int));
    
    //vectors
    float *r,*L;
    int   *z,*m;
    cudaMalloc((void **)&r, H.n_col * sizeof(float));
    cudaMalloc((void **)&L, H.n_col * sizeof(float));
    cudaMalloc((void **)&z, H.n_col * sizeof(int));
    cudaMalloc((void **)&m, H.n_col * sizeof(int));

    //early termination check
    int *d_check;
    cudaMallocManaged((void **)&d_check, 1* sizeof(int));
    *d_check=0;

    //load inital data to device
    cudaMemcpy( m   , recv_codeword , H.n_col       * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( dH  , H.A[0]        , H.n_elements  * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( dHi , H.A[1]        , (H.n_row+1)   * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //printf(" memory initialization time:%3.3f \n",time*1000);
    cudaEventRecord(start, 0);
    tmememory=time;
#endif

    //kernel 0:
    GPU_sparse_apriori_probabilities<<<cw_blocks, threads_per_block>>>(H.n_col, init_prob, m, r, L ,z);
    cudaDeviceSynchronize();

#ifdef DEBUG
       
        error1 = cudaGetLastError();
        printf("Error on kernel 0 %s\n", cudaGetErrorString(error1));
        printf("initialization:\n");
        printf("recv_codeword:[");
        for(int i=0;i<H.n_col;i++){
            printf("%d,",recv_codeword[i]);
        }
        printf("]\n\n");

        printf("m:[");
        cudaMemcpy(codeword_decoded,m,H.n_col*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i=0;i<H.n_col;i++){
            printf("%d,",codeword_decoded[i]);
        }
        printf("]\n\n");

        printf("dH:[");
        cudaMemcpy(index_debug_print,dH,H.n_elements*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i=0;i<H.n_elements;i++){
            printf("%d,",index_debug_print[i]);
        }
        printf("]\n\n");

        printf("dHi:[");
        cudaMemcpy(index_debug_print,dHi,H.n_elements*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i=0;i<H.n_row+1;i++){
            printf("%d,",index_debug_print[i]);
        }
        printf("]\n\n");

        printf("kernel 0:\n");
        cudaMemcpy(vector_debug_print,L,H.n_col*sizeof(float),cudaMemcpyDeviceToHost);
        printf("L:[");
        for(int i=0;i<H.n_col;i++){
            printf("%f,",vector_debug_print[i]);
        }
        printf("]\n\n");

        cudaMemcpy(vector_debug_print,r,H.n_col*sizeof(float),cudaMemcpyDeviceToHost);
        printf("r:[");
        for(int i=0;i<H.n_col;i++){
            printf("%f,",vector_debug_print[i]);
        }
        printf("]\n\n");
#endif
#ifdef TIMES
    //kernel 0 / initialization timings
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //printf(" initialization time:%3.3f\n",time*1000);
    cudaEventRecord(start, 0);
    k0=time;
#endif

    //iterative portion
    int try_n;
    for (try_n = 0; try_n<MAX_ITERATIONS; try_n++){


#ifdef TIMES
        //kernel 1 timings start
        cudaEventRecord(start2, 0);
        //printf("iteration number %d:\n",try_n);
#endif
        //kernel 1:
        GPU_sparse_row_wise<<<rw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, dHi, E, L, z, d_check);
        cudaDeviceSynchronize();
#ifdef DEBUG
        cudaDeviceSynchronize();
        error1 = cudaGetLastError();
        printf("Error in GPU_sparse_row_wise %s\n", cudaGetErrorString(error1));
        printf("iteration nº%d\n",try_n);

        cudaMemcpy(matrix_debug_print,E,H.n_elements*sizeof(float),cudaMemcpyDeviceToHost);
        //this is a print vector
        printf("E:[");
        for(int i=0;i<H.n_elements;i++){
            printf("%f,",matrix_debug_print[i]);
        }
        printf("]\n");

        cudaMemcpy(codeword_decoded,z,H.n_col*sizeof(int),cudaMemcpyDeviceToHost);
        printf("z:[");
        for(int i =0;i<H.n_col;i++)
            printf("%d ",codeword_decoded[i]);
            printf("]\n");

        printf("check:%d\n",*d_check);

        
#endif
#ifdef TIMES
        //kernel 1 timings stop
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime(&time, start2, stop2);
        //printf("    kernel 1 time:%3.3f\n",time*1000);
        k1+=time;
        //kernel 2 timings start
        cudaEventRecord(start2, 0);
#endif
        //early termination (computing is done on kernel 1)
        if (*d_check==0){
#ifdef VERBOSE
            printf("solution was found!\n");
#endif
            break;
        }
        //set early termination to occur
        *d_check=0;


        //kernel 2:
        GPU_sparse_column_wise<<<cw_blocks, threads_per_block>>>(H.n_elements, H.n_col, dH, E, r, L, z);

        cudaDeviceSynchronize();


#ifdef DEBUG          
        error2 = cudaGetLastError();
        printf("Error in GPU_sparse_Column_wise %s\n\n", cudaGetErrorString(error2));

#endif
#ifdef TIMES
        //kernel 2 timings stop
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime(&time, start2, stop2);
        //printf("    kernel 2 time:%3.3f\n",time*1000);
        k2+=time;
#endif
    }

    //get results from the device
    cudaMemcpy(codeword_decoded,z,H.n_col*sizeof(int),cudaMemcpyDeviceToHost);

#ifdef TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    //printf(" %d iterations time:%3.3f \n",try_n,time*1000);
    //this average time may be imperfect due to the problem of adding a small float to a large one 
    //but for arround 200 iteration it shouldn't be a massive issue
    //printf(" average times k1: %f k2:%f\n",k1/try_n*1000,k2/try_n*1000);
    //printf(" %ld",(clock_end-clock_start));
    cudaEventRecord(start, 0);

    //this should be written to a log but when I add a file the program has a weird error
    //printf(log,"%f\t%f\t%f\t%f\t\n",    tmememory,k0,k1/try_n*1000,k2/try_n*1000);
    printf("%f\t%f\t%f\t%f\t%d\n",    tmememory*1000,k0*1000,k1*1000,k2*1000,try_n);
#endif

    return try_n;
}

/*HARD CODED EXAMPLE
int main(int argc, char *argv[]){
    pchk H;
    H.n_row=3;
    H.n_col=6;
    H.n_elements=10;
    H.type = 1;
    H.A   = (int **) malloc(2 *sizeof(int *));
    H.A[0]= (int *)  malloc(10*sizeof(int  ));
    H.A[1]= (int *)  malloc(2 *sizeof(int  ));

    int He[10] = {0,1,3,1,2,4,0,1,2,5};
    int Hi[4]  = {0,3,6,10};

    for(int i=0;i<10;i++)
        H.A[0][i]=He[i];
    for(int i=0;i<4;i++)
        H.A[1][i]=Hi[i];

    int recv_codeword[6] = {1,0,0,0,0,0};
    int codeword_decoded[6] = {0,0,0,0,0,0};
    float error_rate = 0.005;

    GPU_sparse_decode(H,recv_codeword,codeword_decoded,error_rate);

    return 0;
}
*/
