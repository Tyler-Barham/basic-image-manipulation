#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QPixmap>
#include <QImage>
#include <QDebug>

#include "imageprocessor.h"

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
        void on_horizontalSlider_valueChanged( int value );
        void on_buttonReload_clicked();
        void on_buttonProcess_clicked();
        void updateImage( QPixmap newImg );

    signals:
        void thresholdImage( int threshold );

    private:
        Ui::MainWindow *ui;
        ImageProcessor *imgProcessor;
        QThread *imgThread;
        QImage origImg;
        int threshold;
};

#endif // MAINWINDOW_H
