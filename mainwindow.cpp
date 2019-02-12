#include "mainwindow.h"
#include "ui_mainwindow.h"

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

    // Set up the image processor
    imgProcessor = new ImageProcessor( origImg );
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
    emit thresholdImage( threshold );
}

void MainWindow::updateImage( QPixmap newImg )
{
    ui->contentImage->setPixmap( newImg );
}
