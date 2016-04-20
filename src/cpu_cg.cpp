#include <iostream>
#include "CycleTimer.h"
#include "serial_cg.h"
int M = 0, N = 0, nz = 0, *I, *J;
float *val;
const float tol = 1e-5f;
const int max_iter = 10000;
/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (float)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (float)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}

int main(int argc, char **argv)
{

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    M = N = 1048576;
    // M = N = 1048;
    nz = (N-2)*3 + 4;
    // nz = 100;
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (float *)malloc(sizeof(float)*nz);
    double* val_double = (double *)malloc(sizeof(double)*nz);
    genTridiag(I, J, val, N, nz);

    // cusp::csr_matrix<int, float, cusp::host_memory> A(M, N, nz);

    // for (int i = 0; i < N + 1; i++) {
    //     A.row_offsets[i] = I[i];
    // }
    for (int i = 0; i < nz; i++) {
        // A.column_indices[i] = J[i];
        // A.values[i] = val[i];
        val_double[i] = val[i];
    }

    // double cusparse_time = cusparse_cg();
    // double cusp_time = cusp_cg(A);
    double serial_time = serial_cg(I, J, val_double, N);

    free(I);
    free(J);
    free(val);
    free(val_double);
    // printf("cusparse_time: %f\ncusp_time: %f\n", cusparse_time, cusp_time);
}