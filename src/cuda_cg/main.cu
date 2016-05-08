#include "main.h"
#include "CycleTimer.h"

int32_t main(int32_t argc, char* argv[]) {
    // Check the number of parameters
    if (argc < 2) {
        // Tell the user how to run the program
        std::cerr << "Usage: N " << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    setup(N);

    double startTime = CycleTimer::currentSeconds();
    multi_kernel(cuda_vectorX);
    axpby();
    rsold = inner_prod(cuda_r, cuda_r, numRows);
    size_t numBytes = numRows * sizeof(float);
    cudaMemcpy(cuda_p, cuda_r, numBytes, cudaMemcpyDeviceToDevice);

    uint32_t i;
    for (i = 0; i < numRows; i++) {
        multi_kernel(cuda_p);
        alpha = rsold / inner_prod(cuda_vectorY, cuda_r, numRows);
        axpy(cuda_p, cuda_vectorX, alpha);
        axpy(cuda_vectorY, cuda_r, -alpha);
        rsnew = inner_prod(cuda_r, cuda_r, numRows);
        printf("Iteration %d: %f\n", i, rsnew);
        if (rsnew < 1e-10)
        break;
        axby(cuda_r, cuda_p, rsnew / rsold);
        rsold = rsnew;
    }
    printf("Total iteration: %d\n", i);
    double time_cost = CycleTimer::currentSeconds() - startTime;
    printf("Total time: %.3f ms\n", time_cost * 1000.f);

    //store();
    freekernel();

    return 0;
}

void setup(int N) {
    int deviceCount = 0;
    string name;
    cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);

        numThreadsPerBlock = deviceProps.maxThreadsPerBlock;
        numThreadBlocks = deviceProps.multiProcessorCount * 
            (deviceProps.maxThreadsPerMultiProcessor / numThreadsPerBlock);

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    numRows = numCols = N;
    numValues = (numRows - 2) * 3 + 4;
    row_offsets = new uint32_t[numRows + 1];
    column_indices = new uint32_t[numValues];
    values = new float[numValues];

    genTridiag(row_offsets, column_indices, values, numRows, numValues);
    vectorX = new float[numRows]();
    vectorY = new float[numRows]();
    b = new float[numRows];
    for (uint32_t i = 0; i < numRows; i++)
        b[i] = 1.0;
    r = new float[numRows]();
    p = new float[numRows]();

    cudaMalloc(&cuda_RowCounter, sizeof(uint32_t));
    cudaMemcpyToSymbol(cuda_NumRows, &numRows, sizeof(uint32_t));
    cudaMemcpyToSymbol(cuda_NumCols, &numCols, sizeof(uint32_t));
    size_t numBytes;
    numBytes = (numRows + 1) * sizeof(uint32_t);
    cudaMalloc(&cuda_rowOffsets, numBytes);
    cudaMemcpy(cuda_rowOffsets, row_offsets, numBytes, cudaMemcpyHostToDevice);

    numBytes = numValues * sizeof(uint32_t);
    cudaMalloc(&cuda_colIndex, numBytes);
    cudaMemcpy(cuda_colIndex, column_indices, numBytes, cudaMemcpyHostToDevice);

    numBytes = numValues * sizeof(float);
    cudaMalloc(&cuda_values, numBytes);
    cudaMemcpy(cuda_values, values, numBytes, cudaMemcpyHostToDevice);

    numBytes = numCols * sizeof(float);
    cudaMalloc(&cuda_vectorX, numBytes);
    cudaMemcpy(cuda_vectorX, vectorX, numBytes, cudaMemcpyHostToDevice);

    numBytes = numRows * sizeof(float);
    cudaMalloc(&cuda_vectorY, numBytes);
    cudaMemcpy(cuda_vectorY, vectorY, numBytes, cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_b, numBytes);
    cudaMemcpy(cuda_b, b, numBytes, cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_r, numBytes);
    cudaMemcpy(cuda_r, r, numBytes, cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_p, numBytes);
    cudaMemcpy(cuda_p, p, numBytes, cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_input_vector, numBytes);
    cudaMalloc(&cuda_output_vector, 1024*sizeof(float));
}

void multi_kernel(float *x) {
    cudaMemset(cuda_RowCounter, 0, sizeof(uint32_t));
    int32_t meanElementsPerRow = (int32_t) rint(
        (double)numValues / numRows);

    if (meanElementsPerRow <= 2) {
        mvDynamicWarp<2, MAX_THREAD_PER_BLOCK / 2>
            <<<numThreadBlocks, numThreadsPerBlock>>>(
            cuda_rowOffsets, cuda_colIndex, cuda_values,
            x, cuda_vectorY, cuda_RowCounter);
    }
    else if (meanElementsPerRow <= 4) {
        mvDynamicWarp<4, MAX_THREAD_PER_BLOCK / 4>
            <<<numThreadBlocks, numThreadsPerBlock>>>(
            cuda_rowOffsets, cuda_colIndex, cuda_values,
            x, cuda_vectorY, cuda_RowCounter);
    }
    else if (meanElementsPerRow <= 64) {
        mvDynamicWarp<8, MAX_THREAD_PER_BLOCK / 8>
            <<<numThreadBlocks, numThreadsPerBlock>>>(
            cuda_rowOffsets, cuda_colIndex, cuda_values,
            x, cuda_vectorY, cuda_RowCounter);
    }
    else {
        mvDynamicWarp<32, MAX_THREAD_PER_BLOCK / 32>
            <<<numThreadBlocks, numThreadsPerBlock>>>(
            cuda_rowOffsets, cuda_colIndex, cuda_values,
            x, cuda_vectorY, cuda_RowCounter);
    }
    cudaError_t err = cudaThreadSynchronize();
    if (err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(err));
}

void freekernel() {
    cudaFree(cuda_rowOffsets);
    cudaFree(cuda_colIndex);
    cudaFree(cuda_values);
    cudaFree(cuda_vectorX);
    cudaFree(cuda_vectorY);
    cudaFree(cuda_b);
    cudaFree(cuda_r);
    cudaFree(cuda_p);
}

void store() {
    uint32_t numBytes = numRows * sizeof(float);
    cudaMemcpy(vectorX, cuda_vectorX, numBytes, cudaMemcpyDeviceToHost);
    for (uint32_t i = 0; i < numRows; i++) {
        printf("%f\n", vectorX[i]);
    }
}

template <uint32_t THREADS_PER_VECTOR, uint32_t MAX_NUM_VECTORS_PER_BLOCK>
__global__ void mvDynamicWarp(const uint32_t* __restrict cuda_rowOffsets, 
    const uint32_t* __restrict cuda_colIndex, const float* __restrict cuda_values,
    const float* __restrict cuda_vectorX, float* cuda_vectorY, uint32_t* __restrict RowCounter) 
{
    uint32_t i;
    float sum;
    uint32_t row;
    uint32_t rowStart, rowEnd;
    const uint32_t laneId = threadIdx.x % THREADS_PER_VECTOR; /*lane index in the vector*/
    const uint32_t vectorId = threadIdx.x / THREADS_PER_VECTOR; /*vector index in the thread block*/
    const uint32_t warpLaneId = threadIdx.x & 31;   /*lane index in the warp*/
    const uint32_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;  /*vector index in the warp*/

    __shared__ volatile uint32_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

    /*get the row index*/
    if (warpLaneId == 0) {
        row = atomicAdd(RowCounter, 32 / THREADS_PER_VECTOR);
    }
    /*broadcast the value to other threads in the same warp and compute the row index of each vector*/
    row = __shfl(row, 0) + warpVectorId;

    /*check the row range*/
    while (row < cuda_NumRows) {

        /*use two threads to fetch the row offset*/
        if (laneId < 2) {
            space[vectorId][laneId] = cuda_rowOffsets[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        /*there are non-zero elements in the current row*/
        sum = 0;
        /*compute dot product*/
        if (THREADS_PER_VECTOR == 32) {

            /*ensure aligned memory access*/
            i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

            /*process the unaligned part*/
            if (i >= rowStart && i < rowEnd) {
                sum += cuda_values[i] * cuda_vectorX[cuda_colIndex[i]];
            }

                /*process the aligned part*/
            for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += cuda_values[i] * cuda_vectorX[cuda_colIndex[i]];
            }
        } else {
            /*regardless of the global memory access alignment*/
            for (i = rowStart + laneId; i < rowEnd; i +=
                    THREADS_PER_VECTOR) {
                sum += cuda_values[i] * cuda_vectorX[cuda_colIndex[i]];
            }
        }
        /*intra-vector reduction*/
        for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
            sum += __shfl_down(sum, i, THREADS_PER_VECTOR);
        }

        /*save the results and get a new row*/
        if (laneId == 0) {
            /*save the results*/
            cuda_vectorY[row] = sum;
        }

        /*get a new row index*/
        if(warpLaneId == 0){
            row = atomicAdd(RowCounter, 32 / THREADS_PER_VECTOR);
        }
        /*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
        row = __shfl(row, 0) + warpVectorId;

    }/*while*/
}

void axpby() {
    int blocks = divup(numRows, numThreadsPerBlock);
    axpbykernel<<<blocks, numThreadsPerBlock>>>
                          (cuda_b, cuda_vectorY, cuda_r);
    cudaError_t err = cudaThreadSynchronize();
    if (err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(err));
}

__global__ void axpbykernel(const float* __restrict cuda_b, const float* __restrict cuda_vectorY,
                            float* cuda_r)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < cuda_NumRows)
        cuda_r[index] = cuda_b[index] - cuda_vectorY[index];
}

void axpy(float* p, float* x, float alpha) {
    int blocks = divup(numRows, numThreadsPerBlock);
    axpykernel<<<blocks, numThreadsPerBlock>>>
                         (p, x, alpha);
    cudaError_t err = cudaThreadSynchronize();
    if (err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(err));
}

__global__ void axpykernel(const float* __restrict p, float* x,
                           float alpha)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < cuda_NumRows)
        x[index] = x[index] + alpha * p[index];
}

void axby(float* p, float* x, float alpha) {
    int blocks = divup(numRows, numThreadsPerBlock);
    axbykernel<<<blocks, numThreadsPerBlock>>>
                         (p, x, alpha);
    cudaError_t err = cudaThreadSynchronize();
    if (err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(err));
}

__global__ void axbykernel(const float* __restrict p, float* x,
                           float alpha)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < cuda_NumRows)
        x[index] = p[index] + alpha * x[index];
}



/* reduce within a warp, result stored in the first thread */
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 32/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

/* reduce within a block */
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared [1024 / 32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    /* each warp performs partial reduction */
    val = warpReduceSum(val);
    /* first lane of each warp writes its val to shared mem */
    if (lane == 0) {
        shared[wid] = val;
    }
    /* wait for all warp shuffles finish */
    __syncthreads();
    /* only threads from first warp of the block load shared partial results */
    val = (threadIdx.x < blockDim.x/32) ? shared[lane] : 0;
    /* first warp perform final reduction of partial results of all warps */
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

/* reduce within a device */
__global__ void deviceReduceKernel(float* input, float* output, int numRows) {
    float sum = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < numRows; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

__global__ void inner_prod_kernel(float* vector1, float* vector2, float* cuda_input_vector) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < cuda_NumRows) {
        cuda_input_vector[index] = vector1[index] * vector2[index];
    }
}

float inner_prod(float *vector1, float *vector2, uint32_t numRows) {
    float inner_product;
    /* phase 1: multiplication */
    int blocks = divup(numRows, numThreadsPerBlock);
    inner_prod_kernel<<<blocks, numThreadsPerBlock>>>(vector1, vector2, cuda_input_vector);
    
    /* phase 2: cross-block device reduction */
    int threads = 1024;
    blocks = min((numRows + threads - 1) / threads, 1024); // guarantee second reduction can be done within one block

    deviceReduceKernel<<<blocks, threads>>>(cuda_input_vector, cuda_output_vector, numRows);
    deviceReduceKernel<<<1, 1024>>>(cuda_output_vector, cuda_output_vector, blocks);

    cudaMemcpy(&inner_product, cuda_output_vector, sizeof(float), cudaMemcpyDeviceToHost);

    // printf("inner_product = %f\n", inner_product);

    return inner_product;
}

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(uint32_t *I, uint32_t *J, float *val, uint32_t N, uint32_t nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (float)rand()/RAND_MAX + 10.0f;
    val[1] = (float)rand()/RAND_MAX;
    int start;

    for (uint32_t i = 1; i < N; i++)
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

