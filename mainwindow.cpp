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
    ui->contentImage->setPixmap( origImg );
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
    ui->contentImage->setPixmap( origImg );
}

void MainWindow::on_buttonProcess_clicked()
{
    QImage img = origImg.toImage();
    int height = img.height();
    int width = img.width();
    int* imgArr = new int[ width * height ];

    // Store grayscale pixmap into int[]
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            imgArr[ y * width + x ] = qGray( img.pixel( x, y ) );
        }
    }

    // Get the new image as int[]
    int *newImg = computeMask( threshold, imgArr, width, height );

    // Set background pixels to black
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            if( newImg[ y * width + x ] == 0 )
            {
                img.setPixelColor( x, y, QColor( "black" ) );
            }
        }
    }

    // Convert image to pixmap, then display
    QPixmap tempPixmap;
    tempPixmap.convertFromImage( img );
    ui->contentImage->setPixmap( tempPixmap );
}
