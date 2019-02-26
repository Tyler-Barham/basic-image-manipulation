#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <stdio.h>

extern "C" void SetupImageProcessor( cv::Mat image );
extern "C" void DestroyImageProcessor();
extern "C" cv::Mat computeThreshold( int threshold, cv::Mat image );
extern "C" cv::Mat computeEdges( cv::Mat image );

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

    // Use grid-stride loop to ensure all elements are processed
    for( int xIndex = index; xIndex < width * height; xIndex += stride )
    {
        // Pixel density multiplied by pixel location
        const int pid = 3 * xIndex;

        // RGB values of the pixel
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
    const int stride = blockDim.x * gridDim.x;

    // Use grid-stride loop to ensure all elements are processed
    for( int xIndex = index; xIndex < width * height; xIndex += stride )
    {
        // Pixel density multiplied by pixel location
        const int pid = 3 * xIndex;

        // RGB values of the pixel
        const unsigned int curr_blue = imageArray[ pid ];
        const unsigned int curr_green = imageArray[ pid + 1 ];
        const unsigned int curr_red = imageArray[ pid + 2 ];

        // Calculate the grayscale color of the pixel
        const int gray = ( ( curr_red * 11 ) + ( curr_green * 16 ) + ( curr_blue * 5 ) ) / 32;

        // Detect if the pixel is an outline (if so, no need to perform calulation)
        bool isOutline = false;
        if( curr_red == 255 && curr_green == 0 && curr_blue == 0 )
        {
            isOutline = true;
        }

        // If this is a pixel with color && not a previous edge
        if( gray != 0 && !isOutline )
        {
            // Have not yet detected edges
            bool edge = false;

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

            // Number of items in the array
            int neighborsLength = sizeof( neighbors ) / sizeof( int );

            // For each number
            for( int i = 0; i < neighborsLength; i++ )
            {
                // Neighbor location mulitplied by pixel density
                const int neighborID = neighbors[ i ] * 3;

                // If out of range
                if( ( neighborID < 0 ) || ( ( neighborID + 2 ) > ( width * height ) ) )
                {
                    continue;
                }

                // RGB values of the pixel
                const unsigned int blue = imageArray[ neighborID ];
                const unsigned int green = imageArray[ neighborID + 1 ];
                const unsigned int red = imageArray[ neighborID + 2 ];

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
                // Change pixel to red
                imageArray[ pid ] = ( unsigned char ) 0;
                imageArray[ pid + 1 ] = ( unsigned char ) 0;
                imageArray[ pid + 2 ] = ( unsigned char ) 255;
            }
        }
    }
}

int main(int argc, char **argv)
{
    cv::Mat origImg = cv::imread( "/home/tyler/Documents/qt-projects/image_manipulation/resources/testImage.png", CV_LOAD_IMAGE_COLOR );
    cv::Mat newImg = origImg.clone();

    SetupImageProcessor( newImg );

    newImg = computeThreshold( 20, origImg );

    newImg = computeEdges( newImg );

    cv::imshow( "Edges", newImg );
    cv::waitKey( 5000 );
    cv::destroyAllWindows();

    DestroyImageProcessor();

    return 0;
}

void SetupImageProcessor( cv::Mat image )
{
    width = image.cols;
    height = image.rows;
    imageBytes = image.step[0] * image.rows; //strlen( ( char* )image.data );
    blockSize = 1024;
    gridSize = ceil( ( width * height ) + blockSize - 1 ) / blockSize;
    // Allocate device accessible memory to the imageArray
    cudaMalloc<unsigned char>( &imageArray, imageBytes );

    // Image = CV_8UC3 = 8Bit - Unsigned int - 3 channel
}

void DestroyImageProcessor()
{
    // Free cuda allocated memory
    cudaFree( imageArray );
}

cv::Mat computeThreshold( int threshold, cv::Mat image )
{
    cudaMemcpy( imageArray, image.ptr(), imageBytes, cudaMemcpyHostToDevice );

    // Perform thresholding on the image
    applyThreshold<<<gridSize, blockSize>>>( imageArray, threshold, width, height );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy cuda array back to the image
    cudaMemcpy( image.ptr(), imageArray, imageBytes, cudaMemcpyDeviceToHost );

    // Return the updated image
    return image;
}

cv::Mat computeEdges( cv::Mat image )
{
    cudaMemcpy( imageArray, image.ptr(), imageBytes, cudaMemcpyHostToDevice );

    // Perform thresholding on the image
    applyEdgeDetection<<<gridSize, blockSize>>>( imageArray, width, height );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy cuda array back to the image
    cudaMemcpy( image.ptr(), imageArray, imageBytes, cudaMemcpyDeviceToHost );

    // Return the updated image
    return image;
}

// TODO: Make edge detection cover entire screen (currently top 1/3 only)
