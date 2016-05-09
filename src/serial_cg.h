#define TOL 1e-5f
#define MAX_ITER 1000
#define TEST_ROUND 2


void matrix_vector(int num_rows, int *ptr, int *indices, double *data, double *x, double *y);
void vector_subtract(int size, double *vector1, double *vector2, double *result);
void vector_add(int size, double *vector1, double *vector2, double *result);
double vector_vector(int size, double *vector1, double *vector2);
void vector_scalar(int size, double *vector, double scalar, double *result);
double serial_cg(int* ptr, int* indices, double* data, int N);

