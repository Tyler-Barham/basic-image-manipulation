#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

extern "C"
cv::Mat computeThreshold( int threshold, cv::Mat image );
cv::Mat computeEdges( cv::Mat image );

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
        const int pid = yIndex * step + ( 3 * xIndex );

        // BGR values of the pixel
        const unsigned char blue = array[ pid ];
        const unsigned char green = array[ pid + 1 ];
        const unsigned char red = array[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const float gray = ( ( red * 11.0f ) + ( green * 16.0f ) + ( blue * 5.0f ) ) / 32.0f;

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

__global__ void applyEdgeDetection( unsigned char* array,
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
        const int pid = yIndex * step + ( 3 * xIndex );

        // BGR values of the pixel
        const unsigned char red = array[ pid ];
        const unsigned char green = array[ pid + 1 ];
        const unsigned char blue = array[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const float gray = ( ( red * 11.0f ) + ( green * 16.0f ) + ( blue * 5.0f ) ) / 32.0f;

        // If this is a pixel with color
        if( gray != 0 )
        {
            // Have not yet detected edges
            bool edge = false;

            // Location of neighboring pixels in image
            const int neighbors[] = { ( yIndex + 1 ) * step + ( 3 * ( xIndex + 1 ) ),
                                      ( yIndex + 1 ) * step + ( 3 * ( xIndex - 1 ) ),
                                      ( yIndex + 1 ) * step + ( 3 * xIndex ),
                                      ( yIndex - 1 ) * step + ( 3 * ( xIndex + 1 ) ),
                                      ( yIndex - 1 ) * step + ( 3 * ( xIndex - 1 ) ),
                                      ( yIndex - 1 ) * step + ( 3 * xIndex ),
                                        yIndex * step + ( 3 * ( xIndex + 1 ) ),
                                        yIndex * step + ( 3 * ( xIndex - 1 ) ) };

            // Number of items in the array
            int neighborsLength = sizeof( neighbors ) / sizeof( int );

            // For each number
            for( int i = 0; i < neighborsLength; i++ )
            {
                const int currPix = neighbors[ i ];

                // If out of range
                if( ( currPix < 0 ) || ( ( currPix + 2 ) > ( width * height ) ) )
                {
                    continue;
                }

                // BGR values of the pixel
                const unsigned char red = array[ currPix ];
                const unsigned char green = array[ currPix + 1 ];
                const unsigned char blue = array[ currPix + 2 ];

                // If neighbor is black, this is an edge
                if( red == 0 && green == 0 && blue == 0 )
                {
                    edge = true;
                    break;
                }
            }

            // if this was an edge, change the color
            if( edge )
            {
                // Change pixel to black
                array[ pid ] = static_cast<unsigned char>( 255.0f );
                array[ pid + 1 ] = static_cast<unsigned char>( 0.0f );
                array[ pid + 2 ] = static_cast<unsigned char>( 0.0f );
            }
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
    const dim3 block( 16, 16 );

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

cv::Mat computeEdges( cv::Mat image )
{
    // Calculate the total number of bytes
    const int bytes = image.step * image.rows;

    // Malloc and Memcpy the image to a cuda array
    unsigned char *array;
    cudaMallocManaged( &array, bytes );
    cudaMemcpy( array, image.ptr(), bytes, cudaMemcpyHostToDevice );

    // Specify a reasonable block size
    const dim3 block( 16, 16 );

    // Calculate grid size to cover the whole image
    const dim3 grid( ( image.cols + block.x - 1 ) / block.x, ( image.rows + block.y - 1 ) / block.y );

    // Perform thresholding on the image
    applyEdgeDetection<<<grid, block>>>( array, image.cols, image.rows, image.step );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy cuda array back to the image
    cudaMemcpy( image.ptr(), array, bytes, cudaMemcpyDeviceToHost );

    // Free cuda malloc'd memory
    cudaFree( array );

    // Return the updated image
    return image;
}

// TODO: Make edge detection cover entire screen (currently top 1/3 only)
