#-------------------------------------------------
#
# Project created by QtCreator 2019-02-12T09:11:33
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = image_manipulation
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

DESTDIR = $${PWD}/build

CUDA_HEADERS = $$files( include/*.cuda.h )
CUDA_SOURCES = $$files( src/*.cu )

# CUDA settings
CUDA_SDK = "/usr/local/cuda" # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda" # Path to cuda toolkit install

SYSTEM_NAME = unix # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64 # '32' or '64', depending on your system
CUDA_ARCH = sm_21 # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math -g -G

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/

# Add the necessary libraries
CUDA_LIBS = \
    -lcuda \
    -lcudart

OPENCV_LIBS = \
    -L/opt/opencv/lib \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgproc

# include paths
INCLUDEPATH += \
    $$CUDA_DIR/include \
    /opt/opencv/include \
    /opt/Qt/include \
    $${PWD}/include

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

LIBS += \
    $$CUDA_LIBS \
    $$OPENCV_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $${DESTDIR}/${QMAKE_FILE_BASE}.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG \
                                          $$NVCC_OPTIONS \
                                          $$CUDA_INC \
                                          $$LIBS \
                                          --machine $$SYSTEM_TYPE \
                                          -arch=$$CUDA_ARCH \
                                          -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    cuda_d.variable_out = OBJECTS
    QMAKE_EXTRA_COMPILERS += cuda_d
} else {
    # Release mode
    cuda.input = $$CUDA_SOURCES
    cuda.output = $${DESTDIR}/${QMAKE_FILE_BASE}.o
    cuda.commands = $${CUDA_DIR}/bin/nvcc $$NVCC_OPTIONS \
                                          $$CUDA_INC \
                                          $$LIBS \
                                          --machine $$SYSTEM_TYPE \
                                          -arch=$$CUDA_ARCH \
                                          -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    cuda_d.variable_out = OBJECTS
    QMAKE_EXTRA_COMPILERS += cuda
}

SOURCES += $$files( src/*.cpp )
HEADERS += $$files( include/*.h )
FORMS += $$files( ui/*.ui )
RESOURCES += $$files( resources/*.qrc )
