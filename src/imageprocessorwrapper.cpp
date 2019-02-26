#include "imageprocessorwrapper.h"
#include "imageprocessor.cuda.h"

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
