#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C"
int* computeMask( int threshold, int imgArr[], int width, int height );


__global__ void applyMask( int threshold, int length, int *cudaImgArr )
{
    // Determine where in the loop to start
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Determine how quickly to step over the loop
    int stride = blockDim.x * gridDim.x;

    // Calculate if the pixel is part of the background
    for( int i = index; i < length; i += stride )
    {
        if( cudaImgArr[ i ] < threshold )
        {
            cudaImgArr[ i ] = 0;
        }
    }
}


int* computeMask( int threshold, int imgArr[], int width, int height )
{   
    int length = width * height; // Length of the array
    int blockSize = 256; // Number of threads to use per block
    int numBlocks = (length + blockSize - 1) / blockSize; // Number of blocks
    const int bytes = length * sizeof( int ); // Number of bytes to be allocated
    int *cudaImgArr;

    // Allocate GPU memory send the image argument to device accessible memory
    cudaMallocManaged( &cudaImgArr, bytes);
    cudaMemcpy( cudaImgArr, imgArr, bytes, cudaMemcpyHostToDevice );

    // Perform calculations on GPU
    applyMask<<<numBlocks, blockSize>>>( threshold, length, cudaImgArr );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Transfer the cuda var back to the host
    cudaMemcpy( imgArr, cudaImgArr, bytes, cudaMemcpyDeviceToHost );

    // Free the memory
    cudaFree( cudaImgArr );

    return imgArr;
}
