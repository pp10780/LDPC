# Notes

## Encoding

Operation:
 - Matrix Multiplication

 - How to respresent message data
 - How to obtain the generator matrix

 MATRIX H (PARITY CHECK MATRIX) - m x n where m is the number of constraints and n is length of the codewords
HcT = 0 where cT is the transpose of the codeword

 MATRIX G (GENERATOR MATRIX) - k x n where k is the length of the message bits and n is the length of the codeword
c = uG where u is the message bits

- How to obtain G from H?
H = [A, I_{n-k}]
G = [I_k, AT]

## Decoding

Operation:
 - Sum-product decoding

RECEIVED VECTOR: y
CROSSOVER PROBABILITY: p
A PRIORI PROBABILITIES VECTOR: r 

r_i = log (p/(1-p)) if y_i = 1,
    = log ((1-p)/p) if y_1 = 0

MESSAGES FROM BIT NODE TO CHECK NODE MATRIX: M_{j,i} where j number of check nodes and i is the number of i nodes

M_{j,i} = sum_{j' belongs to A and j' =/= j}(E_{j',i} + r_i)

Initialization: M_{j,i} = r_i (ATTENTION CHECK THIS INITIALIZATION I BELIEVE THERE IS A CONDITION MISSING WHERE, FOR EACH COLUMN, ONLY THE ROWS THAT HAVE ONE (WHERE THAT BIT NODE IS USED) THE r is substituted and not the whole column as it is presented)

EXTRINSIC PROBABILITIES MATRIX: E

E_{j,i} = (page 32 eq 2.7)

L is the total LLR for each iteration and is given by

L_i = r_i + sum(E_{i,j}) for every checknode j that contains i

z is the current codeword after each iteration where z_i = 1 if L_i < 0
                                                     z_i = 0 if L_i > 0

- How to test if the obtained value is or isn't a code word

s = zH' and s must be a vector with only 0's

H' is HT

## Matrix representation

As matrizes neste momento têm este formato num ficheiro txt:
    >1st line: number_rows number_columns type
    >2nd-end : matrix 
    
Como o valor para as matrizes the paridade são 1 ou 0 só é necessário guardar os valores dos indices
Isto é inefeciente e têm que ser alterado já está no TODO
a maneira melhor seria guardar isto exatamente como está guardado dentro do programa (binario).

Dentro do programa em si as matrizes estão gruardadas numa estrutura pchk definida no defs
a estrutura é basicamente o que está no ficheiro



## Example for encoding

[1 0 0 1 0 1]
[0 1 0 1 1 1]
[0 0 1 0 1 1]


[0 0 0] = [0 0 0 0 0 0]
[0 0 1] = [0 0 1 0 1 1]
[0 1 0] = [0 1 0 1 1 1]
[0 1 1] = [0 1 1 1 0 0]
[1 0 0] = [1 0 0 1 0 1]
[1 0 1] = [1 0 1 1 1 0]
[1 1 0] = [1 1 0 0 1 0]
[1 1 1] = [1 1 1 0 0 1]

Ativa as colunas em que a mensagem está a 1 e dá-lhes xor.

Implementação do Mackday por exemplo para o [0 1 1]:
pega na segunda linha e copia-a para a codeword
[0 1 0 1 1 1]
depois pega na terceira linha e da xor das duas
[0 1 0 1 1 1] xor [0 0 1 0 1 1] = [0 1 1 1 0 0]


## TODO 

-makefile GPU a funcionar
-test if it works
-get iteration times for different matrices




gcc -std=c99 -g -c -Wall src/main.c -o obj/main.o
gcc -std=c99 -g -c -Wall src/decoding.c -o obj/decoding.o
gcc -std=c99 -g -c -Wall src/encoding.c -o obj/encoding.o
gcc -std=c99 -g -c -Wall src/display_variables.c -o obj/display_variables.o
gcc -std=c99 -g -c -Wall src/storage.c -o obj/storage.o
gcc -std=c99 -g -c -Wall src/sparse_decoding.c -o obj/sparse_decoding.o
nvcc -O3 -m64 -c src/GPU_decoding.cu -o obj/GPU_decoding.o
gcc -g obj/main.o obj/decoding.o obj/encoding.o obj/display_variables.o obj/storage.o obj/sparse_decoding.o obj/GPU_decoding.o -o bin/ldpc  -lm
obj/GPU_decoding.o: In function `early_termination(int, int, int*, int*, int*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xe7): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x140): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `GPU_row_wise(int, int, int*, float*, float*, float*, int*, int*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x269): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x2c6): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `GPU_column_wise(int, int, int*, float*, float*, float*, int*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x3d9): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x436): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `GPU_apriori_probabilities(int, float, int*, float*, float*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x519): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x570): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `GPU_decode(Pchk, int*, int*, float)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x756): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x766): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x776): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x788): undefined reference to `cudaMemset'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x7a1): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x7b1): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x7c1): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x7d1): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x7e3): undefined reference to `cudaMalloc'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x7fa): undefined reference to `cudaMemset'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x814): undefined reference to `cudaMemcpy'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x830): undefined reference to `cudaMemcpy'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x8a0): undefined reference to `__cudaPushCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x9a5): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x9b2): undefined reference to `cudaDeviceSynchronize'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x9cc): undefined reference to `cudaDeviceSynchronize'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x9ec): undefined reference to `cudaMemset'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xa39): undefined reference to `__cudaPushCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xb95): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xba2): undefined reference to `cudaDeviceSynchronize'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xbf0): undefined reference to `__cudaPushCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xd3a): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xdae): undefined reference to `cudaLaunchKernel'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xdb3): undefined reference to `cudaDeviceSynchronize'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xdda): undefined reference to `cudaMemcpy'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xe55): undefined reference to `cudaLaunchKernel'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xea4): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `__device_stub__Z25GPU_apriori_probabilitiesifPiPfS0_(int, float, int*, float*, float*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xf61): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0xfac): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `__device_stub__Z12GPU_row_wiseiiPiPfS0_S0_S_S_(int, int, int*, float*, float*, float*, int*, int*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x10a1): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x10ec): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `__device_stub__Z15GPU_column_wiseiiPiPfS0_S0_S_(int, int, int*, float*, float*, float*, int*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x11d1): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x121c): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `__device_stub__Z17early_terminationiiPiS_S_(int, int, int*, int*, int*)':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x12df): undefined reference to `__cudaPopCallConfiguration'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x1324): undefined reference to `cudaLaunchKernel'
obj/GPU_decoding.o: In function `__cudaUnregisterBinaryUtil()':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text+0x13): undefined reference to `__cudaUnregisterFatBinary'
obj/GPU_decoding.o: In function `__sti____cudaRegisterAll()':
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text.startup+0xb): undefined reference to `__cudaRegisterFatBinary'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text.startup+0x64): undefined reference to `__cudaRegisterFunction'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text.startup+0xa5): undefined reference to `__cudaRegisterFunction'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text.startup+0xe6): undefined reference to `__cudaRegisterFunction'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text.startup+0x127): undefined reference to `__cudaRegisterFunction'
tmpxft_00003603_00000000-5_GPU_decoding.cudafe1.cpp:(.text.startup+0x133): undefined reference to `__cudaRegisterFatBinaryEnd'
collect2: error: ld returned 1 exit status
