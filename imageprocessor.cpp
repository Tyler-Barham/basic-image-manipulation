#include "imageprocessor.h"

extern "C"
cv::Mat computeThreshold( int threshold, cv::Mat image );

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
