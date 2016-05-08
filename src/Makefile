all: test

test: test.cu serial_cg.cpp
	nvcc -I./include -g -o test test.cu serial_cg.cpp -lcublas_static -lcusparse_static -lculibos

clean:
	rm -f *.o test
