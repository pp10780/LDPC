#ifndef DEF
    #define DEF

    #define MAX_ITERATIONS 5

    #define CODEWORD_LEN 6 // N this cannot be fixed
    #define MESSAGE_LEN 3  // K this cannot be fixed
    #define NUM_CHECKS 4   // M this cannot be fixed

    #define BSC_MODE 0
    #define AWGN_MODE 1

    #define CURR_MODE BSC_MODE

    #define BSC_ERROR_RATE 0.2

    typedef struct Pchk{
        int n_row;
        int n_col;
        int type;
        int **A;
    } pchk;

    //TODO: number of threads per block
    //#define THREADS_PER_BLOCK ?

    #define DEBUG
#endif