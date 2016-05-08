all: gpu_test

gpu_test: main.cu
	nvcc --cudart static -O3 -I./../include -arch sm_35 -DHAVE_SM_35 -rdc true --ptxas-options=-v -Xcompiler -funroll-loops -Xcompiler -Wall -o gpu_test main.cu -lm

clean:
	rm -f *.o gpu_test