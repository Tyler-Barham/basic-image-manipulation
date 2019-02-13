#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cstring>

extern "C"
cv::Mat computeThreshold( int threshold, cv::Mat image );

__global__ void applyThreshold( int threshold,
                           unsigned char* array,
                           int width,
                           int height,
                           int step )
{
    // 2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    // Only valid threads perform memory I/O
    if( ( xIndex < width ) && ( yIndex < height ) )
    {
        // Location of pixel in image
        const int pid = yIndex * step + (3 * xIndex);

        // BGR values of the pixel
        const unsigned char blue = array[ pid ];
        const unsigned char green = array[ pid + 1 ];
        const unsigned char red = array[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const float gray = ( ( red * 11.0f ) + ( green * 16.0f ) + ( blue * 5.0f) ) / 32.0f;

        // If pixel is darker than the threshold
        if( gray < threshold )
        {
            // Change pixel to black
            array[ pid ] = static_cast<unsigned char>( 0.0f );
            array[ pid + 1 ] = static_cast<unsigned char>( 0.0f );
            array[ pid + 2 ] = static_cast<unsigned char>( 0.0f );
        }
    }
}

cv::Mat computeThreshold( int threshold, cv::Mat image )
{
    // Calculate the total number of bytes
    const int bytes = image.step * image.rows;

    // Malloc and Memcpy the image to a cuda array
    unsigned char *array;
    cudaMallocManaged( &array, bytes);
    cudaMemcpy( array, image.ptr(), bytes, cudaMemcpyHostToDevice );

    // Specify a reasonable block size
    const dim3 block(16,16);

    // Calculate grid size to cover the whole image
    const dim3 grid( ( image.cols + block.x - 1 ) / block.x, ( image.rows + block.y - 1 ) / block.y );

    // Perform thresholding on the image
    applyThreshold<<<grid, block>>>( threshold, array, image.cols, image.rows, image.step );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy cuda array back to the image
    cudaMemcpy( image.ptr(), array, bytes, cudaMemcpyDeviceToHost );

    // Free cuda malloc'd memory
    cudaFree( array );

    // Return the updated image
    return image;
}
