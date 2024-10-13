#ifndef DEF
    #define DEF

    #define MAX_ITERATIONS 50

    #define BSC_MODE 0
    #define AWGN_MODE 1

    #define CURR_MODE BSC_MODE

    #define DEFAULT_ERROR_RATE 0.2
    #define DEFAULT_MAX_ERRORS -1 //this means no max

    typedef struct Pchk{
        int n_row;
        int n_col;
        int n_elements;
        int type;
        int **A;
    } pchk;

    //#define GPU
    //#define DEBUG
    #define TIMES
    #define RESULT
#endif
