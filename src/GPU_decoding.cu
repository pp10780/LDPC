#include <stdio.h>
#include <time.h>


extern "C" {
#include "GPU_decoding.h"
#include "defs.h"
}

//kernel 0: innit -> compute r and Li from m
__global__ void GPU_apriori_probabilities(int n_col, float llr_i , int *m, float *r, float *L){
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
__global__ void GPU_row_wise(int n_row, int n_col, int *H, float *M, float* E, float *L, int *z, int *d_check){

    float LE = 1; //row value used to compute E
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = n_col*j;
    int check=0;
    
    if(j > n_row)
        return;

    //do full row for M [first recursion]
    for(int i = 0 ; i<n_col; i++){
        if(H[row_start + i]!=0){
            //early termination check (this is being done in parallel)
            check ^= z[i];

            float M_val = L[i] - E[row_start + i];

            //store row value
            LE *= tanh(M_val/2);
            //writing result in global memory
            M[row_start + i] = M_val;
        }
    }

    //do full row for E [second recursion]
    for(int i = 0 ; i<n_col; i++){
        if(H[row_start + i]!=0){
            //exclude corresponding element from row -> this is going back to global memory which min sum doesn't have to (BAD!)
            float p = LE/(tanh(M[row_start + i]/2) );
            E[row_start + i] = log((1+p)/(1-p));
        }
    }

    //this is probably very bad maybe do a reduction?
    if(check == 1)
        *d_check=0;
}

//kernel 2: column wise -> compute L and z from E and r
__global__ void GPU_column_wise(int n_row, int n_col, int *H, float* E, float* r,float *L, int *z){
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    float L_val;//only write to global memory in the end

    if(i > n_col)
        return;
    
    L_val=r[i];
    //go through E column wise
    for(int j=0; j<n_row; j++){
        if (H[j*n_col + i]!=0){
            L_val+=E[j*n_col + i];
        }
    }

    L[i] = L_val;
    z[i] = (L_val < 0) ? 1 : 0;;
}

//this is very inneficient has of right now has it is not the focus and I didn't understant how they're doing it in the other one
//kernel 3: early termination -> see if word is a success
__global__ void early_termination(int n_row, int n_col, int *H, int *z, int *d_check){
    int j = (blockIdx.x * blockDim.x + threadIdx.x);
    int row_start = n_col*j;

    if(j > n_row)
        return;

    //TODO: this is extremelly inneficient!
    int check=0;

    //TODO: this could be merged into kernel 1
    //going row wise
    for(int i = 0 ; i<n_col; i++){
        if(H[ row_start*j + i] == 1)
            check ^= z[i];
    }

    //this is probably very bad maybe do a reduction?
    if(check == 1)
        *d_check=0;
}

// Function to decode the message
extern "C"
void GPU_decode(pchk H, int *recv_codeword, int *codeword_decoded, float error_rate){
#ifdef TIMES
    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    //initialize device memory
    int check;
    //Couldn't find a work arround to memcpy being the only way to insert this into GPU 
    int* H_mem;
    H_mem = (int *)calloc(H.n_row*H.n_col,sizeof(int));
    for(int j=0;j< H.n_row;j++){
        for(int i=0;i<H.n_col;i++)
            H_mem[j*H.n_col+i]=H.A[j][i];
    }
    //TODO: this is temporary I still need to calculate the number of blocks required and set the number of threads per block in defs
    int threads_per_block=32;//CL_NV_DEVICE_WARP_SIZE
    //TODO: make this better
    int rw_blocks=H.n_col/threads_per_block + 1;
    int cw_blocks=H.n_row/threads_per_block + 1;
    int MAX_ITERATION = 10;

    float init_prob=log((1 - error_rate)/error_rate);

    //decoding matrix
    int *dH;
    cudaMalloc((void **)&dH, H.n_row * H.n_col * sizeof(int));

    //computation matrices
    float *M,*E;
    cudaMalloc((void **)&M, H.n_row * H.n_col * sizeof(float));
    cudaMalloc((void **)&E, H.n_row * H.n_col * sizeof(float));
    //E needs to be set at 0 at the start
    cudaMemset(M,0,H.n_row * H.n_col * sizeof(int));

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
    cudaMemcpy( dH, H_mem, H.n_row*H.n_col * sizeof(int), cudaMemcpyHostToDevice);

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
        GPU_row_wise<<<rw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, M, E, L, z, d_check);
        cudaDeviceSynchronize();

        //kernel 2:
        GPU_column_wise<<<cw_blocks, threads_per_block>>>(H.n_row, H.n_col, dH, E, r, L, z);

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