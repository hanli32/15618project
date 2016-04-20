nvcc -g -o test test.cu serial_cg.cpp -lcublas_static -lcusparse_static -lculibos
