#ifndef DISPLAY_VARIABLES_H
#define DISPLAY_VARIABLES_H

#include <stdio.h>
#include <stdlib.h>
#include "decoding.h"
#include "defs.h"

// Function to print a vector int
void print_vector_int(int vector[], int len);

// Function to print a vector float
void print_vector_float(float vector[], int len);

// Function to print a matrix int
void print_matrix_int(int** matrix, int rows, int cols);

//Function to print the structure parity check(pchk)
void print_parity_check(pchk mat);

//function to print sparse floating point matrices
void print_sparse_float(pchk index,float *mat);

// Function to print a matrix float
void print_matrix_float(float** matrix, int rows, int cols);

// Function to create a list of all possible codewords
//void possible_codeword_list(int ***list);

// Function to check all possible codewords
//void check_possible_codewords(pchk H);

#endif