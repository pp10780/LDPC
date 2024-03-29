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
void print_vector_float(float vector[], int len)
{
    printf("[ ");
    for (int i = 0; i < len; i++)
        printf("%f ", vector[i]);
    printf("]\n");
}

// Function to print a matrix int
void print_matrix_int(int** matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        printf("[ ");
        for (int j = 0; j < cols; j++)
            printf("%d ", matrix[i][j]);
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

// Function to create a list of all possible codewords
void possible_codeword_list(int ***list)
{
    *list = (int**)malloc(64 * sizeof(int*));
    for (int i = 0; i < 64; i++)
    {
        (*list)[i] = (int*)malloc(CODEWORD_LEN * sizeof(int));
    }

    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < CODEWORD_LEN; j++)
        {
            (*list)[i][CODEWORD_LEN - 1 - j] = (i >> j) & 1;
        }
    }
}

// Function to check all possible codewords
void check_possible_codewords(int H[NUM_CHECKS][CODEWORD_LEN])
{
    int *codeword_decoded;
    int **possible_codeword_l;
    possible_codeword_list(&possible_codeword_l);
    
    codeword_decoded = (int*)malloc(CODEWORD_LEN * sizeof(int));
    int check = 0;
    for (int i = 0; i < 64; i++)
    {
        check = check_codeword(H, possible_codeword_l[i]);
            printf("[ ");
            for (int j = 0; j < CODEWORD_LEN; j++)
                printf("%d ", possible_codeword_l[i][j]);
            printf("] -> ");
        if (check == 1)
        {
            printf("Valid codeword\n");
        }
        else
        { 
            decode(H, possible_codeword_l[i], codeword_decoded);
            if (codeword_decoded == NULL)
            {
                printf("Not a valid codeword\n");
                continue;
            }
            else
            {
                print_vector_int(codeword_decoded, CODEWORD_LEN);   
            }
        }
    }

    free(codeword_decoded);
    for (int i = 0; i < 64; i++)
    {
        free(possible_codeword_l[i]);
    }
    free(possible_codeword_l);
}