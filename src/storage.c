#include "storage.h"

void **get_matrix_from_file(pchk *matrix,char *filename){
    FILE *f = fopen (filename,"r");
    int i=0;

    //Open file to read
    if(f==NULL){
        printf("couldn't open matrix file %s\n",filename);
        exit(1);
    }    

    //matrix info
    fread(&(matrix->n_row),sizeof(int),1,f);
    fread(&(matrix->n_col),sizeof(int),1,f);
    fread(&(matrix->type),sizeof(int),1,f);

    //A
    matrix->A = (int**)malloc(matrix->n_row*sizeof(int*));
    if(matrix->type ==0){
        //normal
        for(int r=0;r<matrix->n_row;r++){
            matrix->A[r] = (int*)malloc(matrix->n_col*sizeof(int));
            fread(matrix->A[r],sizeof(int),matrix->n_col,f);
        }
    }
    else{
        //sparse
        for(int r=0;r<matrix->n_row;r++){
            fread(&i,sizeof(int),1,f);
            matrix->A[r] = (int*)malloc((i+1)*sizeof(int));
            matrix->A[r][0] = i;
            fread(&(matrix->A[r][1]),sizeof(int),i,f);
        }
    }
    fclose(f);
    return NULL;
}