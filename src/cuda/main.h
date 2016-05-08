#ifndef __MAIN_H__
#define __MAIN_H__

#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <thrust/inner_product.h>
using namespace std;

#define MAX_THREAD_PER_BLOCK    1024

int32_t numThreadsPerBlock;
int32_t numThreadBlocks;

uint32_t* cuda_RowCounter;

uint32_t* row_offsets;
uint32_t* column_indices;
float* values;
uint32_t numRows;
uint32_t numCols;
uint32_t numValues;

float rsold = 0.f;
float rsnew = 0.f;
float alpha = 0.f;

float* vectorX;
float* vectorY;

float* b;
float* r;
float* p;

uint32_t* cuda_rowOffsets;
uint32_t* cuda_colIndex;
float* cuda_values;
float* cuda_vectorY;
float* cuda_vectorX;
float* cuda_b;
float* cuda_r;
float* cuda_p;
//
float* cuda_input_vector;
float* cuda_output_vector;

__constant__ uint32_t cuda_NumRows;
__constant__ uint32_t cuda_NumCols;

void setup();
void multi_kernel(float* x);
void freekernel();
void store();
void axpby();
void axpy(float* p, float* x, float alpha);
void axby(float* p, float* x, float alpha);
void genTridiag(uint32_t *I, uint32_t *J, float *val, uint32_t N, uint32_t nz);
float inner_prod(float *vector1, float *vector2, uint32_t numRows);

template <uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void mvDynamicWarp(const uint32_t* __restrict cuda_rowOffsets, 
    const uint32_t* __restrict cuda_colIndex, const float* __restrict cuda_values,
    const float* __restrict cuda_vectorX, float* cuda_vectorY, uint32_t* __restrict RowCounter);

__global__ void axpbykernel(const float* __restrict cuda_b, const float* __restrict cuda_vectorY,
        float* cuda_r);

__global__ void axpykernel(const float* __restrict p, float* x,
        float alpha);

__global__ void axbykernel(const float* __restrict p, float* x,
        float alpha);

__host__ __device__ int divup(uint32_t x, uint32_t y) { return x / y + (x % y ? 1 : 0); }

#endif


