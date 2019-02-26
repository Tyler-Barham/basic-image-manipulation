#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "imageprocessor.cuda.h"

unsigned char *imageArray;
int imageBytes;
int width;
int height;
int blockSize;
int gridSize;

__global__ void applyThreshold( unsigned char *imageArray, int threshold, const int width, const int height )
{
    // Index of current thread
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride amount
    const int stride = blockDim.x * gridDim.x;

    // Image channel count
    const int channels = 3;

    // Use grid-stride loop to ensure all elements are processed
    for( int xIndex = index; xIndex < width * height; xIndex += stride )
    {
        // Pixel location multiplied by channels to get correct array index
        const int pid = xIndex * channels;

        // BGR values of the pixel
        const unsigned int blue = imageArray[ pid ];
        const unsigned int green = imageArray[ pid + 1 ];
        const unsigned int red = imageArray[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const int gray = ( ( red * 11 ) + ( green * 16 ) + ( blue * 5 ) ) / 32;

        // Detect if the pixel is an outline (if so, pixel won't be thresholded)
        bool isOutline = false;
        if( red == 255 && green == 0 && blue == 0 )
        {
            isOutline = true;
        }

        // If pixel is darker than the threshold && not a previous edge
        if( gray < threshold && !isOutline )
        {
            // Change pixel to black
            imageArray[ pid ] = ( unsigned char ) 0;
            imageArray[ pid + 1 ] = ( unsigned char ) 0;
            imageArray[ pid + 2 ] = ( unsigned char ) 0;
        }
    }
}

__global__ void applyEdgeDetection( unsigned char *imageArray, const int width, const int height )
{
    // Index of current thread
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Image stride amount
    const int stride = blockDim.x * gridDim.x;

    // Image channel count
    const int channels = 3;

    // Use grid-stride loop to ensure all elements are processed
    for( int xIndex = index; xIndex < width * height; xIndex += stride )
    {
        // Pixel location multiplied by channels to get correct array index
        const int pid = xIndex * channels;

        // BGR values of the pixel
        const unsigned int curr_blue = imageArray[ pid ];
        const unsigned int curr_green = imageArray[ pid + 1 ];
        const unsigned int curr_red = imageArray[ pid + 2 ];

        // If this is a black pixel
        if( curr_red == 0 && curr_green == 0 && curr_blue == 0 )
        {
            // Location of neighboring pixels in image ( +-1, +-width )
            const int neighbors[] = { ( xIndex + 1 ),
                                      ( xIndex + 1 + width ),
                                      ( xIndex + 1 - width ),
                                      ( xIndex + width ),
                                      ( xIndex - 1 ),
                                      ( xIndex - width ),
                                      ( xIndex - 1 + width ),
                                      ( xIndex - 1 - width )
                                    };

            // Number of neighbors in the array
            int neighborsLength = sizeof( neighbors ) / sizeof( int );

            // For each neighbor
            for( int nIdx = 0; nIdx < neighborsLength; nIdx++ )
            {
                // Neighbor pixel location multiplied by channels to get correct array index
                const int neighborPid = neighbors[ nIdx ] * channels;

                // If not out of range
                if( ( neighborPid > 0 ) && ( neighborPid < ( width * height * channels ) ) )
                {
                    // BGR values of the pixel
                    const unsigned int blue = imageArray[ neighborPid ];
                    const unsigned int green = imageArray[ neighborPid + 1 ];
                    const unsigned int red = imageArray[ neighborPid + 2 ];

                    // Check if the neighbor is a colored pixel
                    if( red != 0 || green != 0 || blue != 0 )
                    {
                        // Detect if the colored neighbor is actually an outline and skip if so
                        if( red == 255 && green == 0 && blue == 0 )
                        {
                            continue;
                        }

                        // Change this pixel to a red one (indicating an outline)
                        imageArray[ pid ] = ( unsigned char ) 0;
                        imageArray[ pid + 1 ] = ( unsigned char ) 0;
                        imageArray[ pid + 2 ] = ( unsigned char ) 255;
                    }
                }
            }
        }
    }
}

// Main - used when testing this file alone (i.e. without QT)
/*
int main(int argc, char **argv)
{
    cv::Mat origImg = cv::imread( "/home/tyler/Documents/qt-projects/image_manipulation/resources/testImage.png", CV_LOAD_IMAGE_COLOR );
    cv::Mat newImg = origImg.clone();

    SetupImageProcessor( newImg );

    newImg = computeThreshold( 20, origImg );

    newImg = computeEdges( newImg );

    cv::imshow( "Edges", newImg );
    cv::waitKey( 10000 );
    cv::destroyAllWindows();

    DestroyImageProcessor();

    return 0;
}
*/

void SetupImageProcessor( cv::Mat image )
{
    // Setup global vars
    width = image.cols;
    height = image.rows;
    imageBytes = image.step[0] * image.rows;
    blockSize = 1024;
    gridSize = ceil( ( width * height ) + blockSize - 1 ) / blockSize;

    // Allocate device accessible memory to the imageArray
    cudaMalloc<unsigned char>( &imageArray, imageBytes );

    // image.type() = 16 = CV_8UC3 = 8Bit, unsigned int, 3 channel
}

void DestroyImageProcessor()
{
    // Free cuda allocated memory
    cudaFree( imageArray );
}

cv::Mat computeThreshold( int threshold, cv::Mat image )
{
    // Copy the image data into a device accessible array
    cudaMemcpy( imageArray, image.ptr(), imageBytes, cudaMemcpyHostToDevice );

    // Perform thresholding on the image
    applyThreshold<<<gridSize, blockSize>>>( imageArray, threshold, width, height );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy device array back to the image
    cudaMemcpy( image.ptr(), imageArray, imageBytes, cudaMemcpyDeviceToHost );

    // Return the updated image
    return image;
}

cv::Mat computeEdges( cv::Mat image )
{
    // Copy the image data into a device accessible array
    cudaMemcpy( imageArray, image.ptr(), imageBytes, cudaMemcpyHostToDevice );

    // Perform edge detection on the image
    applyEdgeDetection<<<gridSize, blockSize>>>( imageArray, width, height );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy device array back to the image
    cudaMemcpy( image.ptr(), imageArray, imageBytes, cudaMemcpyDeviceToHost );

    // Return the updated image
    return image;
}
