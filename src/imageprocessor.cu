#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

extern "C" void SetupImageProcessor( cv::Mat image );
extern "C" void DestroyImageProcessor();
extern "C" cv::Mat computeThreshold( int threshold, cv::Mat image );
extern "C" cv::Mat computeEdges( cv::Mat image );

unsigned char *imageArray;
int imageBytes;
int width;
int height;
int step;
int block;
int grid;

__global__ void applyThreshold( unsigned char *imageArray, int threshold, const int width, const int height, const int step )
{
    // 2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Only valid threads perform memory I/O
    if( xIndex < width + ( width * blockIdx.x ) )
    {
        // Location of pixel in image
        const int pid = 3 * xIndex;

        // RGB values of the pixel
        const unsigned char red = imageArray[ pid ];
        const unsigned char green = imageArray[ pid + 1 ];
        const unsigned char blue = imageArray[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const float gray = ( ( red * 11.0f ) + ( green * 16.0f ) + ( blue * 5.0f ) ) / 32.0f;

        // If pixel is darker than the threshold
        if( gray < threshold )
        {
            // Change pixel to black
            imageArray[ pid ] = static_cast<unsigned char>( 0.0f );
            imageArray[ pid + 1 ] = static_cast<unsigned char>( 0.0f );
            imageArray[ pid + 2 ] = static_cast<unsigned char>( 0.0f );
        }
    }
}

__global__ void applyEdgeDetection( unsigned char *imageArray, const int width, const int height, const int step )
{
    // 2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Only valid threads perform memory I/O
    if( xIndex < width + ( width * blockIdx.x ) )
    {
        // Location of pixel in image
        const int pid = 3 * xIndex;

        // RGB values of the pixel
        const unsigned char red = imageArray[ pid ];
        const unsigned char green = imageArray[ pid + 1 ];
        const unsigned char blue = imageArray[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const float gray = ( ( red * 11.0f ) + ( green * 16.0f ) + ( blue * 5.0f ) ) / 32.0f;

        // If this is a pixel with color
        if( gray != 0 )
        {
            // Have not yet detected edges
            bool edge = false;

            // Location of neighboring pixels in image ( +-1, +-width )
            const int neighbors[] = { ( xIndex + 1 ) * 3,
                                      ( xIndex + 1 + width ) * 3,
                                      ( xIndex + 1 - width ) * 3,
                                      ( xIndex + width ) * 3,
                                      ( xIndex - 1 ) * 3,
                                      ( xIndex - width ) * 3,
                                      ( xIndex - 1 + width ) * 3,
                                      ( xIndex - 1 - width ) * 3
                                    };

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
                const unsigned char red = imageArray[ currPix ];
                const unsigned char green = imageArray[ currPix + 1 ];
                const unsigned char blue = imageArray[ currPix + 2 ];

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
                imageArray[ pid ] = static_cast<unsigned char>( 255.0f );
                imageArray[ pid + 1 ] = static_cast<unsigned char>( 0.0f );
                imageArray[ pid + 2 ] = static_cast<unsigned char>( 0.0f );
            }
        }
    }
}

void SetupImageProcessor( cv::Mat image )
{
    width = image.cols;
    height = image.rows;
    step = image.step;
    imageBytes = step * height;
    block = 256;
    grid = ( ( width * height ) + block - 1 ) / block;
    // Allocate device accessible memory to the imageArray
    cudaMallocManaged( &imageArray, imageBytes );
}

void DestroyImageProcessor()
{
    // Free cuda allocated memory
    cudaFree( imageArray );
}

cv::Mat computeThreshold( int threshold, cv::Mat image )
{
    cudaMemcpy( imageArray, image.data, imageBytes, cudaMemcpyHostToDevice );

    // Perform thresholding on the image
    applyThreshold<<<grid, block>>>( imageArray, threshold, width, height, step );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy cuda array back to the image
    cudaMemcpy( image.ptr(), imageArray, imageBytes, cudaMemcpyDeviceToHost );

    // Return the updated image
    return image;
}

cv::Mat computeEdges( cv::Mat image )
{
    cudaMemcpy( imageArray, image.data, imageBytes, cudaMemcpyHostToDevice );

    // Perform thresholding on the image
    applyEdgeDetection<<<grid, block>>>( imageArray, width, height, step );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy cuda array back to the image
    cudaMemcpy( image.ptr(), imageArray, imageBytes, cudaMemcpyDeviceToHost );

    // Return the updated image
    return image;
}

// TODO: Make edge detection cover entire screen (currently top 1/3 only)
