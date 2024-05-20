#include "display_variables.h"

// Function to print a vector int
void print_vector_int(int vector[], int len)
{
    printf("[ ");
    for (int i = 0; i < len; i++)
        printf("%d ", vector[i]);
    printf("]\n");
}

// Function to print a vector float
void print_vector_float(float vector[], int len){
    printf("[ ");
    for (int i = 0; i < len; i++)
        printf("%f ", vector[i]);
    printf("]\n");
}

// Function to print a matrix int
void print_matrix_int(int** matrix, int rows, int cols){
    for (int i = 0; i < rows; i++)
    {
        printf("[ ");
        for (int j = 0; j < cols; j++)
            printf("%d ", matrix[i][j]);
        printf("]\n");
    }
}

//Function to print the structure parity check(pchk)
void print_parity_check(pchk mat){
    if(mat.type == 0){
        //normal
        for (int i = 0; i < mat.n_row; i++)
        {
            printf("[ ");
            for (int j = 0; j < mat.n_col; j++)
                printf("%d ", mat.A[i][j]);
            printf("]\n");
        }
    }
    else{
        //sparse
        for (int i = 0; i < mat.n_row; i++){
            printf("[ ");
            for (int j = mat.A[1][i]; j < mat.A[1][i+1]; j++)
                printf("%d ", mat.A[0][j]);
            printf("]\n");
        }
    }
}

//function to print sparse floating point matrices
void print_sparse_float(pchk index,float *mat){
    for (int i = 0; i < index.n_row; i++){
        printf("[ ");
        for (int j = index.A[1][i]; j < index.A[1][i+1]; j++)
            printf("%f ", mat[j]);    
        printf("]\n");
    }   
}

// Function to print a matrix float
void print_matrix_float(float** matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        printf("[ ");
        for (int j = 0; j < cols; j++)
            printf("%f ", matrix[i][j]);
        printf("]\n");
    }
}
