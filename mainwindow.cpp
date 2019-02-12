#include "mainwindow.h"
#include "ui_mainwindow.h"

extern "C"
int* computeMask( int threshold, int imgArr[], int width, int height );

MainWindow::MainWindow( QWidget *parent ) :
    QMainWindow( parent ),
    ui( new Ui::MainWindow ),
    origImg( ":/images/testImage.png" )
{
    ui->setupUi( this );

    // Get the current slider value
    threshold = ui->horizontalSlider->value();

    // Set the label pixmap as the original image
    ui->contentImage->setPixmap( QPixmap::fromImage( origImg ) );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_horizontalSlider_valueChanged( int value )
{
    // Update threshold then update the ui
    threshold = value;
    ui->contentThreshold->setText( QString::number( threshold ) );
}

void MainWindow::on_buttonReload_clicked()
{
    // Set the label pixmap as the original image
    ui->contentImage->setPixmap( QPixmap::fromImage( origImg ) );
}

void MainWindow::on_buttonProcess_clicked()
{
    int width = origImg.width();
    int height = origImg.height();
    QImage newImg( width, height, QImage::Format_RGB16 );
    int *imgArray = new int[ width * height ];

    // Store grayscale pixmap into int[]
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            imgArray[ y * width + x ] = qGray( origImg.pixel( x, y ) );
        }
    }

    // Get the new image as int[]
    imgArray = computeMask( threshold, imgArray, width, height );

    // Set pixels to appropriate color
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int rgb = imgArray[ y * width + x ];
            newImg.setPixelColor( x, y, QColor( rgb, rgb, rgb ) );
        }
    }

    // Display the image
    ui->contentImage->setPixmap( QPixmap::fromImage( newImg ) );

    delete[] imgArray;
}
