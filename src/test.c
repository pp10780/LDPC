#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]){
    FILE *f;
    char filename[100] = "matrices/G1";

    int nrow=3,ncol=6,type=1,nelements=10;
    /*
    //H
    //int E[10] = {0,1,3,1,2,4,0,1,2,5};
    //int r[4]  = {0,3,6,10};

    //G
    int E[10] = {0,3,5,1,3,4,5,2,4,5};
    int r[4]  = {0,3,7,10};

    f = fopen (filename,"w+");

    fwrite(&nrow,sizeof(int),1,f);
    fwrite(&ncol,sizeof(int),1,f);
    fwrite(&nelements,sizeof(int),1,f);

    fwrite(E,sizeof(int),10,f);
    fwrite(r,sizeof(int),4,f);
    
    fclose(f);
    */
    //--------------------------------------------------------------------------------
    //xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    //--------------------------------------------------------------------------------
    f = fopen (filename,"r");
    int i=0;
    int e=0;
    int **A;

    fread(&nrow,sizeof(int),1,f);
    printf("%d ",nrow);
    fread(&ncol,sizeof(int),1,f);
    printf("%d ",ncol);
    fread(&nelements,sizeof(int),1,f);
    printf("%d\n",nelements);
    
    
    if(nelements ==0){
        //normal
        A = (int**)malloc(nrow*sizeof(int*));
        for(int c=0;c<nrow;c++){
            A[c] = (int*)malloc(ncol*sizeof(int));
            fread(A[c],sizeof(int),ncol,f);
            for(int c2=0;c2<ncol;c2++)
                printf("%d ",A[c][c2]);
            printf("\n");
        }
    }
    else{
        //sparse
        A = (int**)malloc(2*sizeof(int*));
        A[0] = (int*)malloc(nelements*sizeof(int));
        A[1] =(int*)malloc(nrow+1*sizeof(int));

        fread(A[0],sizeof(int),nelements,f);
        fread(A[1],sizeof(int),nrow+1,f);
        
        for(int r=0;r<nrow;r++){
            for(int j=A[1][r]; j<A[1][r+1] ;j++)
                printf("%d ",A[0][j]);
            printf("\n");
        }
        
    }


    fclose(f);
}