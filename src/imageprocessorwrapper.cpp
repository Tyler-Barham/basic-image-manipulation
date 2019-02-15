#include "include/imageprocessorwrapper.h"

extern "C" void SetupImageProcessor( cv::Mat image );
extern "C" void DestroyImageProcessor();
extern "C" cv::Mat computeThreshold( int threshold, cv::Mat image );
extern "C" cv::Mat computeEdges( cv::Mat image );

ImageProcessorWrapper::ImageProcessorWrapper( cv::Mat image )
{
    SetupImageProcessor( image );
}

ImageProcessorWrapper::~ImageProcessorWrapper()
{
    DestroyImageProcessor();
}

void ImageProcessorWrapper::startThresholding( int threshold, cv::Mat image )
{
    // Signal that the thresholding is complete with a new cv::Mat from the cuda file
    emit thresholdComplete( computeThreshold( threshold, image ) );
}

void ImageProcessorWrapper::startEdgeDetection(cv::Mat image)
{
    emit edgesComplete( computeEdges( image ) );
}
