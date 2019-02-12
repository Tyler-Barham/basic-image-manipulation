#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <QColor>

class ImageProcessor : public QObject
{
    Q_OBJECT
    public:
        ImageProcessor( QImage origImg );
        ~ImageProcessor();

    signals:
        void thresholdComplete( QPixmap updatedImg );

    public slots:
        void startThresholding( int threshold );

    private:
        QImage origImg;
        int imgWidth;
        int imgHeight;
};

#endif // IMAGEPROCESSOR_H
