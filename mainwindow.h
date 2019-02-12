#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QColor>
#include <QImage>
#include <QPixmap>
#include <QDebug>

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

private:
        Ui::MainWindow *ui;
        QImage origImg;
        int threshold;
};

#endif // MAINWINDOW_H
