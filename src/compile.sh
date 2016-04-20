# nvcc -o test.o -c test.cu
# nvcc -o test test.o serial_cg.o -lcublas_static -lcusparse_static -lculibos
nvcc -g -o test test.cu serial_cg.cpp -lcublas_static -lcusparse_static -lculibos