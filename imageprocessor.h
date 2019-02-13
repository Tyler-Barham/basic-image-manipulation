#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <QColor>
#include <opencv2/opencv.hpp>

class ImageProcessor : public QObject
{
    Q_OBJECT
    public:
        ImageProcessor();
        ~ImageProcessor();

    signals:
        void thresholdComplete( cv::Mat updatedImg );

    public slots:
        void startThresholding( int threshold, cv::Mat image );
};

#endif // IMAGEPROCESSOR_H
