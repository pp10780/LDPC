#include "storage.h"


//DENSE

void dense_free_pchk(pchk mat){
    for(int i=0;i<mat.n_row;i++)
            free(mat.A[i]);
    free(mat.A);
}

void dense_store(pchk pc, char* filename){
    FILE *f;
    f = fopen (filename,"w+");

    //nrow ncol type
    fwrite(&(pc.n_row),sizeof(int),1,f);
    fwrite(&(pc.n_col),sizeof(int),1,f);
    fwrite(&(pc.n_elements),sizeof(int),1,f);
    fwrite(&(pc.type),sizeof(int),1,f);

    //matrix
    for(int row=0; row<pc.n_row; row++)
        fwrite(pc.A[row],sizeof(int),pc.n_col,f);

    fclose(f);
}

//SPARSE

void sparse_free_pchk(pchk mat){
    free(mat.A[0]);
    free(mat.A[1]);
    free(mat.A);
}

void sparse_store(pchk pc, char* filename){
    FILE *f;
    f = fopen (filename,"w+");

    //nrow ncol type
    fwrite(&(pc.n_row),sizeof(int),1,f);
    fwrite(&(pc.n_col),sizeof(int),1,f);
    fwrite(&(pc.n_elements),sizeof(int),1,f);
    fwrite(&(pc.type),sizeof(int),1,f);

    //matrix
    fwrite(pc.A[0],sizeof(int),pc.n_elements,f);
    fwrite(pc.A[1],sizeof(int),pc.n_row+1,f);

    fclose(f);
}

//GENERAL
//this function frees the insides of the structure mat
void free_pchk(pchk mat){
    switch(mat.type){
        case 0://dense
            dense_free_pchk(mat);
            break;
        default://sparse
            sparse_free_pchk(mat);
    }
}

void store(pchk pc, char* filename){
    if(pc.type==0)
        dense_store(pc,filename);
    else{
        sparse_store(pc,filename);
    }
}

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
    fread(&(matrix->n_elements),sizeof(int),1,f);
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
        matrix->A[0] = (int *)malloc(matrix->n_elements*sizeof(int ));
        matrix->A[1] = (int *)malloc((matrix->n_row+1 ) *sizeof(int ));

        fread(matrix->A[0],sizeof(int),matrix->n_elements,f);
        fread(matrix->A[1],sizeof(int),matrix->n_row+1,f);
    }
    fclose(f);
    return NULL;
}