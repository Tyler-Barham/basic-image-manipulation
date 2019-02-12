#include "imageprocessor.h"

extern "C"
int* computeMask( int threshold, int imgArr[], int width, int height );

ImageProcessor::ImageProcessor( QImage origImg )
{
    imgWidth = origImg.width();
    imgHeight = origImg.height();
    this->origImg = origImg;
}

ImageProcessor::~ImageProcessor()
{

}

void ImageProcessor::startThresholding( int threshold )
{
    QImage newImg( imgWidth, imgHeight, QImage::Format_RGB16 );
    int *imgArray = new int[ imgWidth * imgHeight ];

    // Store grayscale pixmap into int[]
    for(int y = 0; y < imgHeight; y++)
    {
        for(int x = 0; x < imgWidth; x++)
        {
            imgArray[ y * imgWidth + x ] = qGray( origImg.pixel( x, y ) );
        }
    }

    // Get the new image as int[]
    imgArray = computeMask( threshold, imgArray, imgWidth, imgHeight );

    // Set pixels to appropriate color
    for(int y = 0; y < imgHeight; y++)
    {
        for(int x = 0; x < imgWidth; x++)
        {
            int rgb = imgArray[ y * imgWidth + x ];
            newImg.setPixelColor( x, y, QColor( rgb, rgb, rgb ) );
        }
    }

    delete[] imgArray;

    emit thresholdComplete( QPixmap::fromImage( newImg ) );
}
