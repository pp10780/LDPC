#include "storage.h"

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
        matrix->A    = (int**)malloc(2               *sizeof(int*));
        matrix->A[0] = (int *)malloc(matrix->type    *sizeof(int ));
        matrix->A[1] = (int *)malloc(matrix->n_row+1 *sizeof(int ));

        
        fread(matrix->A[0],sizeof(int),matrix->type,f);
        fread(matrix->A[1],sizeof(int),matrix->n_row+1,f);
    }
    fclose(f);
    return NULL;
}