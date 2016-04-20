#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "CycleTimer.h"
#include "serial_cg.h"

double serial_cg(int* ptr, int* indices, double* data, int N) 
{
	double* x = (double *) calloc(N, sizeof(double));
	double* b = (double *) calloc(N, sizeof(double));
	for (int i = 0; i < N; i++) {
        b[i] = 1.0;
        x[i] = 0.0;
    }

	int size = N;
	int row = N;
	float tol = TOL;

	double rsold = 0;
	double rsnew = 0;
	double alpha = 0;
	double *ap = (double *) calloc(size, sizeof(double));
	double *ax = (double *) calloc(size, sizeof(double));
	double *r = (double *) calloc(size, sizeof(double));
	double *p = (double *) calloc(size, sizeof(double));
	double *alphap = (double *) calloc(size, sizeof(double));
	double *alphaap = (double *) calloc(size, sizeof(double));
	double *rp = (double *) calloc(size, sizeof(double));

	matrix_vector(row, ptr, indices, data, x, ax);
	vector_subtract(size, b, ax, r);
	memcpy(p, r, size * sizeof(*r));
	rsold = vector_vector(size, r, r);

    double startTime = CycleTimer::currentSeconds();

	for (int i = 0; i < MAX_ITER; i++) {
		matrix_vector(row, ptr, indices, data, p, ap);
		alpha = rsold / (vector_vector(size, p, ap));
		vector_scalar(size, p, alpha, alphap);
		vector_add(size, x, alphap, x);
		vector_scalar(size, ap, alpha, alphaap);
		vector_subtract(size, r, alphaap, r);
		rsnew = vector_vector(size, r, r);

		printf("iteration = %3d, residual = %e\n", i, sqrt(rsnew));

		if (rsnew < tol * tol)
			break;
		vector_scalar(size, p, rsnew / rsold, rp);
		vector_add(size, r, rp, p);
		rsold = rsnew;
	}

	double time_cost = CycleTimer::currentSeconds() - startTime;

	free(ap);
	free(ax);
	free(r);
	free(p);
	free(alphap);
	free(alphaap);
	free(rp);
	free(x);
	free(b);

    return time_cost;
}

void matrix_vector(int num_rows, int *ptr, int *indices, double *data, double *x, double *y) {
	for (int i = 0; i < num_rows; i++) {
		int row_begin = ptr[i];
		int row_end = ptr[i + 1];
		y[i] = 0;
		for (int j = row_begin; j < row_end; j++) {
			y[i] += data[j] * x[indices[j]];
		}
	}
}

void vector_subtract(int size, double *vector1, double *vector2, double *result) {
	for (int i = 0; i < size; i++) {
		result[i] = vector1[i] - vector2[i];
	}
}

void vector_add(int size, double *vector1, double *vector2, double *result) {
	for (int i = 0; i < size; i++) {
		result[i] = vector1[i] + vector2[i];
	}
}

double vector_vector(int size, double *vector1, double *vector2) {
	double result = 0;

	for (int i = 0; i < size; i++) {
		result += vector1[i] * vector2[i];
	}

	return result;
}

void vector_scalar(int size, double *vector, double scalar, double *result) {
	for (int i = 0; i < size; i++) {
		result[i] = vector[i] * scalar;
	}
}
