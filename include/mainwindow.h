#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QThread>
#include <QPixmap>
#include <QImage>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include <QFile>
#include <opencv2/opencv.hpp>

#include "include/imageprocessorwrapper.h"

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

    public:
        explicit MainWindow( QWidget *parent = 0 );
        ~MainWindow();

    private slots:
        void updateImage( cv::Mat newImg );
        void on_horizontalSlider_valueChanged( int value );
        void on_buttonThreshold_clicked();
        void on_buttonEgdes_clicked();
        void on_buttonClear_clicked();

signals:
        void thresholdImage( int threshold, cv::Mat image );
        void detectEdges( cv::Mat image );

    private:
        void updateUIWithCurrImage();

        Ui::MainWindow *ui;
        ImageProcessorWrapper *imgProWrapper;
        QThread *imgThread;
        cv::Mat origImg;
        cv::Mat currImg;
        int threshold;
};

#endif // MAINWINDOW_H
