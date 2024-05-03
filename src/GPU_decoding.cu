#include "decoding.h"

//kernel 0: innit -> compute r and Li from m
__global__ void GPU_apriori_probabilities(int n_col, float llr_i , float *r, float *L){
    //llr_i corresponds to the initial llr that's attributed depending on the channel (-llr_i if == 1) 
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index > n_col)
        return;
    //TODO:this could be fancier by just changing the signal bit according to the data bit 
    float r_val=(m==0) ? llr_i : -llr_i;

    //write to global memory
    r[index] = r_val;
    L[index] = r_val;
}

//kernel 1: row wise -> compute M and "LE" from L and E, then compute E from M and "LE"
__global__ void GPU_row_wise(int n_row, int n_col, int *H, float *M, float* E){

    float LE = 1; //row value used to compute E
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start = n_col*j;
    
    if(j > n_row)
        return;

    //do full row for M [first recursion]
    for(int i = 0 ; i<n_col; i++){
        if(H[row_start + i]!=0){
            float M_val = L[i] - E[row_start + i];

            //store row value
            LE * = tanh(M_val/2);
            //writing result in global memory
            M[row_start + i] = M_val;
        }
    }

    //do full row for E [second recursion]
    for(int i = 0 ; i<n_col; i++){
        if(H[row_start + i]!=0){
            //exclude corresponding element from row -> this is going back to global memory which min sum doesn't have to (BAD!)
            float p = LE/M[row_start + i] ;
            E[row_start + i] = log((1+p)/(1-p));
        }
    }
}

//kernel 2: column wise -> compute L and z from E
__global__ void GPU_column_wise(int n_row, int n_col, float* E,float *L, int *z){
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
/*
__global__ void early_termination(int n_row, int n_col, int *H, int *z, int d_check){
    int j = (blockIdx.x * blockDim.x + threadIdx.x);

    //this is probably very bad
    if(j==0)
        *d_check = 0;

    if(j > n_row)
        return;

    //this is extremelly inneficient!
    int check=0;

    //going row wise
    for(int i = 0 ; i<n_col; i++){
        if(z[i] == 1)
            check ^= H[ + i];
    }

    //this is probably very bad maybe do a reduction?
    if(check ==1)
        *d_check=1;
}
*/

// Function to decode the message
void GPU_decode(pchk H, int *recv_codeword, int *codeword_decoded){
    //initialize device memory
    //decoding matrix
    float *dH;
    cudaMalloc((void **)&dH, H.n_row * H.n_col * sizeof(float));

    //computation matrices
    float *M,*E;
    cudaMalloc((void **)&M, H.n_row * H.n_col * sizeof(float));
    cudaMalloc((void **)&E, H.n_row * H.n_col * sizeof(float));

    //vectors
    float *r,*L;
    int   *z,*dm;
    cudaMalloc((void **)&r, H.n_col * sizeof(float));
    cudaMalloc((void **)&L, H.n_col * sizeof(float));
    cudaMalloc((void **)&z, H.n_col * sizeof(int));
    cudaMalloc((void **)&dm, H.n_col * sizeof(int));

    //load inital data to device
    cudaMemcpy(dm, recv_codeword, n_col * sizeof(int), cudaMemcpyHostToDevice);
    //not very confident in this thing bellow
    for(int i=0;i< H.n_row;i++)
        cudaMemcpy( &(dH[i*H.n_col]), H[i], H.n_col * sizeof(int), cudaMemcpyHostToDevice);

    //kernel 0:
    GPU_apriori_probabilities<<<blocks, THREADS_PER_BLOCK>>>(H.n_col, log((1 - BSC_ERROR_RATE)/BSC_ERROR_RATE) , dr, dL);
    cudaCheckError(cudaDeviceSynchronize());

    for (int try_n = 0; try_n<MAX_ITERATION; try_n++){

        //kernel 1:
        GPU_row_wise<<<blocks, THREADS_PER_BLOCK>>>(H.n_row, H.n_col, dH, dM, dE);
        cudaCheckError(cudaDeviceSynchronize());
        //kernel 2:
        GPU_column_wise<<<blocks, THREADS_PER_BLOCK>>>(H.n_row, H.n_col, dH, dE, dL, dz);

        //(add later) kernel 3
        //early_termination(n_row, n_col, dH, dz, d_check);
        //if done()?
            //break;
        cudaCheckError(cudaDeviceSynchronize());
    }

    return ;
}