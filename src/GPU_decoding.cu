#include "decoding.h"

//kernel 0: innit -> compute r and Li from m
__global__ void ldpc_cnp_kernel(float * dev_llr,
                                float * dev_dt,
                                float * dev_R,
                                int * dev_et,
                                int threadsPerBlock)
{
    //threadId.x
    //blockId.x
}

//kernel 1: row wise -> compute M and "LE" from L and E, then compute E from M and "LE"  

//kernel 2: column wise -> compute L and z from E

//kernel 3: early termination -> see if word is a success

// Function to decode the message
void GPU_decode(pchk H, int *recv_codeword, int *codeword_decoded)
{
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

    //kernel 0

    for (int try_n = 0; try_n<MAX_ITERATION; try_n++){
        //kernel 1
        //sync
        //kernel 2
        //(add later) kernel 3
        //if done()
            //break;
    }

    return ;
}