#include "imageprocessor.h"

extern "C"
cv::Mat computeThreshold( int threshold, cv::Mat image );
cv::Mat computeEdges( cv::Mat image );

ImageProcessor::ImageProcessor()
{

}

ImageProcessor::~ImageProcessor()
{

}

void ImageProcessor::startThresholding( int threshold, cv::Mat image )
{
    // Signal that the thresholding is complete with a new cv::Mat from the cuda file
    emit thresholdComplete( computeThreshold( threshold, image ) );
}

void ImageProcessor::startEdgeDetection(cv::Mat image)
{
    emit edgesComplete( computeEdges( image ) );
}
