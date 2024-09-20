#ifndef DEF
    #define DEF

    #define MAX_ITERATIONS 400

    #define BSC_MODE 0
    #define AWGN_MODE 1

    #define CURR_MODE BSC_MODE

    #define BSC_ERROR_RATE 0.07

    typedef struct Pchk{
        int n_row;
        int n_col;
        int n_elements;
        int type;
        int **A;
    } pchk;

    //#define GPU
    //#define DEBUG
    //#define TIMES
    #define RESULT
#endif