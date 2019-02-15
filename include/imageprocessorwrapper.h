#ifndef IMAGEPROCESSORWRAPPER_H
#define IMAGEPROCESSORWRAPPER_H

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <QColor>
#include <opencv2/opencv.hpp>

class ImageProcessorWrapper : public QObject
{
    Q_OBJECT
    public:
        ImageProcessorWrapper( cv::Mat image );
        ~ImageProcessorWrapper();

    signals:
        void thresholdComplete( cv::Mat updatedImg );
        void edgesComplete( cv::Mat updatedImg );

    public slots:
        void startThresholding( int threshold, cv::Mat image );
        void startEdgeDetection( cv::Mat image );
};

#endif // IMAGEPROCESSORWRAPPER_H
