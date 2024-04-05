#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
    int nrow=3,ncol=6,type=1;
    int A1[6] = {3,0,1,3};
    int A2[6] = {3,1,2,4};
    int A3[6] = {4,0,1,2,5};

    FILE *f;

    f = fopen ("matrices/Hs1","w+");

    fwrite(&nrow,sizeof(int),1,f);
    fwrite(&ncol,sizeof(int),1,f);
    fwrite(&type,sizeof(int),1,f);
    fwrite(A1,sizeof(int),6,f);
    fwrite(A2,sizeof(int),6,f);
    fwrite(A3,sizeof(int),6,f);

    fclose(f);
    
    f = fopen (argv[1],"r");
    int i=0;
    int **A;

    fread(&nrow,sizeof(int),1,f);
    printf("%d ",nrow);
    fread(&ncol,sizeof(int),1,f);
    printf("%d ",ncol);
    fread(&type,sizeof(int),1,f);
    printf("%d\n",type);
    
    A = (int**)malloc(nrow*sizeof(int*));
    if(type ==0){
        //normal
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
        for(int c=0;c<nrow;c++){
            fread(&i,sizeof(int),1,f);
            printf("%d ",i);
            //this is not storing i and it should! -> fixed in the actual read
            A[c] = (int*)malloc(i*sizeof(int));
            fread(A[c],sizeof(int),i,f);
            for(int c2=0;c2<i;c2++)
                printf("%d ",A[c][c2]);
            printf("\n");
        }
    }


    fclose(f);
}