#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <iostream>
#include "CycleTimer.h"
#include "serial_cg.h"

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

/**
 * check correctness of solver solution
 * @param  x 
 * @param  b 
 * @return err
 */
float check_err(float *x, float *b, int N, int* I, int* J, float* val) {
    float rsum, diff, err = 0.0;
    for (int i = 0; i < N; i++) {
        rsum = 0.0;

        for (int j = I[i]; j < I[i + 1]; j++) {
            rsum += val[j] * x[J[j]];
        }

        diff = fabs(rsum - b[i]);

        if (diff > err) {
            err = diff;
        }
    }
    return err;
}

/**
 * generate a random tridiagonal symmetric matrix
 * @param I   offset
 * @param J   column 
 * @param val values
 * @param N   matrix row/col size
 * @param nz  nonzero size
 */
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
/**
 * cuSPARSE version cg solver
 * @return overall duration
 */
double cusparse_cg(int N, int nz, int* I, int* J, float* val) 
{
    float *x;
    float *rhs;
    float a, b, na, r0, r1;
    int *d_col, *d_row;
    float *d_val, *d_x, dot;
    float *d_r, *d_p, *d_Ax;
    int k;
    float alpha, beta, alpham1;

    x = (float *) malloc(sizeof(float) * N);
    rhs = (float *) malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        rhs[i] = 1.0;
        x[i] = 0.0;
    }

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    (cudaMalloc((void **) &d_col, nz * sizeof(int)));
    (cudaMalloc((void **) &d_row, (N + 1) * sizeof(int)));
    (cudaMalloc((void **) &d_val, nz * sizeof(float)));
    (cudaMalloc((void **) &d_x, N * sizeof(float)));
    (cudaMalloc((void **) &d_r, N * sizeof(float)));
    (cudaMalloc((void **) &d_p, N * sizeof(float)));
    (cudaMalloc((void **) &d_Ax, N * sizeof(float)));

    cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, rhs, N * sizeof(float), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.;

    cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x,
                   &beta, d_Ax);

    cublasSaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
    cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    k = 1;

    while (r1 > TOL * TOL && k <= MAX_ITER) {
        if (k > 1) {
            b = r1 / r0;
            cublasStatus = cublasSscal(cublasHandle, N, &b, d_p, 1);
            cublasStatus = cublasSaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
        }
        else {
            cublasStatus = cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col,
                       d_p, &beta, d_Ax);
        cublasStatus = cublasSdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
        a = r1 / dot;

        cublasStatus = cublasSaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
        na = -a;
        cublasStatus = cublasSaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

        r0 = r1;
        cublasStatus = cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
        cudaThreadSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    double time_cost = CycleTimer::currentSeconds() - startTime;

    cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    // float err = check_err(x, rhs);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(x);
    free(rhs);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    // printf("Test Summary:  Error amount = %f\n", err);
    return time_cost;
}

/**
 * CUSP version cg solver
 * @return overall duration
 */
template <class LinearOperator>
double cusp_cg(LinearOperator& A) {

    cusp::csr_matrix<int,float,cusp::device_memory> d_A(A);
    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(d_A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(d_A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = TOL * TOL
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<float> monitor(b, 100, TOL * TOL, 0, true);

    // set preconditioner (identity)
    cusp::identity_operator<float, cusp::device_memory> M(d_A.num_rows, d_A.num_rows);

    double startTime = CycleTimer::currentSeconds();
    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(d_A, x, b, monitor, M);

    double time_cost = CycleTimer::currentSeconds() - startTime;

    return time_cost;
}

int main(int argc, char **argv)
{
    // Check the number of parameters
    if (argc < 2) {
        // Tell the user how to run the program
        std::cerr << "Usage: N " << std::endl;
        return 1;
    }

    int M = 0, N = 0, nz = 0, *I, *J;
    float *val; 

    /* Generate a random tridiagonal symmetric matrix in CSR format */
    M = N = std::atoi(argv[1]);//1048576;
    std::cout<<"============== N = "<< N <<" ====================\n";
    nz = (N-2)*3 + 4;
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (float *)malloc(sizeof(float)*nz);
    double* val_double = (double *)malloc(sizeof(double)*nz);
    genTridiag(I, J, val, N, nz);

    /* Generate CUSP matrix data type */
    cusp::csr_matrix<int, float, cusp::host_memory> A(M, N, nz);
    for (int i = 0; i < N + 1; i++) {
        A.row_offsets[i] = I[i];
    }
    for (int i = 0; i < nz; i++) {
        A.column_indices[i] = J[i];
        A.values[i] = val[i];
        val_double[i] = val[i];
    }

    double cusparse_time = cusparse_cg(N, nz, I, J, val);
    double cusp_time = cusp_cg(A);
    double serial_time = serial_cg(I, J, val_double, N);

    for (int i = 0; i < 5; i++) {
        cusparse_time = std::min(cusparse_time, cusparse_cg(N, nz, I, J, val));
        cusp_time = std::min(cusp_time, cusp_cg(A));
        serial_time = std::min(serial_time, serial_cg(I, J, val_double, N));
    }

    free(I);
    free(J);
    free(val);
    free(val_double);
    printf("cusparse_time: %f\ncusp_time: %f\nserial_time: %f\n", 
        cusparse_time * 1000, cusp_time * 1000, serial_time * 1000);
}
