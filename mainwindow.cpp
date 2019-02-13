#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow( QWidget *parent ) :
    QMainWindow( parent ),
    ui( new Ui::MainWindow )
{
    ui->setupUi( this );

    // Get the current slider value
    threshold = ui->horizontalSlider->value();

    // Create temp dir
    QTemporaryDir tempDir;
    if ( tempDir.isValid() )
    {
        // Create location of temp file
        const QString tempFile = tempDir.path() + "/testImage.png";

        // Store the qt resource in the temp file
        if ( QFile::copy( ":/images/testImage.png", tempFile ) )
        {
            // Use opencv to read the image from the absolute path
            origImg = cv::imread( tempFile.toStdString() );
            currImg = origImg.clone();

            if( !origImg.empty() )
            {
                // Update the UI to contain the image
                updateUIWithCurrImage();
            }
        }
    }

    // Register cv::Mat so that it can be used in signals/slots
    qRegisterMetaType< cv::Mat >("cv::Mat");

    // Set up the image processor
    imgProcessor = new ImageProcessor();
    imgThread = new QThread();
    imgProcessor->moveToThread( imgThread );
    connect( this, &MainWindow::thresholdImage, imgProcessor, &ImageProcessor::startThresholding );
    connect( imgProcessor, &ImageProcessor::thresholdComplete, this, &MainWindow::updateImage );
    imgThread->start();
}

MainWindow::~MainWindow()
{
    delete ui;
    imgProcessor->deleteLater();
    imgThread->deleteLater();
}

void MainWindow::updateUIWithCurrImage()
{
    // Create a QImage from cv::Mat, convert it to a QPixmap, then update the UI
    ui->contentImage->setPixmap( QPixmap::fromImage( QImage( ( unsigned char* ) currImg.data,
                                                     currImg.cols,
                                                     currImg.rows,
                                                     QImage::Format_RGB888 ) ) );
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
    currImg = origImg.clone();
    updateUIWithCurrImage();
}

void MainWindow::on_buttonProcess_clicked()
{
    // Start the thresholding process
    emit thresholdImage( threshold, currImg );
}

void MainWindow::updateImage( cv::Mat newImg )
{
    currImg = newImg;
    // Update the UI with the thresholded image
    updateUIWithCurrImage();
}
