#ifndef CUDA_H
#define CUDA_H

extern "C" void SetupImageProcessor( cv::Mat image );
extern "C" void DestroyImageProcessor();
extern "C" cv::Mat computeThreshold( int threshold, cv::Mat image );
extern "C" cv::Mat computeEdges( cv::Mat image );

#endif // CUDA_H
