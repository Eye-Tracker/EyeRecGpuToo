#ifndef CAMERAWIDGET_H
#define CAMERAWIDGET_H

#include <QMainWindow>
#include <QThread>
#include <QCamera>
#include <QAction>
#include <QMouseEvent>
#include <QPainter>
#include <QFont>
#include <QMessageBox>

#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

#include "Camera.h"
#include "ImageProcessor.h"
#include "EyeImageProcessor.h"
#include "FieldImageProcessor.h"

#include "DataRecorder.h"

#include "Synchronizer.h"

#include "utils.h"

namespace Ui {
class CameraWidget;
}

class CameraWidget : public QMainWindow, InputWidget
{
    Q_OBJECT

public:
    explicit CameraWidget(QString id, ImageProcessor::Type type, QWidget *parent = 0);
    ~CameraWidget();

signals:
    void setCamera(QCameraInfo cameraInfo);
    void newROI(QPointF sROI, QPointF eROI);
    void newData(EyeData data);
    void newData(FieldData data);
    void newClick(Timestamp,QPoint,QSize);

public slots:
    void preview(Timestamp t, const cv::Mat &frame);
    void preview(EyeData data);
    void preview(FieldData data);
    void preview(const DataTuple &data);
    void options(QAction* action);
    void noCamera(QString msg);

    void startRecording();
    void stopRecording();

    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

    void validatePoint(QPointF &point);

    void collectCameraCalibration(Timestamp t, cv::Mat frame);

private:
    QString id;
    ImageProcessor::Type type;

    Ui::CameraWidget *ui;

    Camera *camera;
    QThread *cameraThread;

    ImageProcessor *imageProcessor;
    QThread *processorThread;

    DataRecorderThread *recorder;
    QThread *recorderThread;

    QActionGroup *optionsGroup;
    QAction *optionAction;

    std::list<int> dt;
    Timestamp lastTimestamp;
    void updateFrameRate(Timestamp t);

    QPointF sROI, eROI;
    bool settingROI;

    Timestamp lastUpdate;
    Timestamp updateIntervalMs;
    Timestamp maxAgeMs;
    bool shouldUpdate(Timestamp t);
    bool isDataRecent(Timestamp t);

    QImage previewImage(const cv::Mat &frame);

    QElapsedTimer cameraCalTimer;
    bool calibratingCamera;
    std::vector< std::vector<cv::Point2f> > imagePoints;
    int collectionIntervalMs;
    int collectionCount;
    cv::Size patternSize;
    double squareSizeMM;


    // Drawing functions
    double rw, rh;
    double refPx;
    QFont font;
    void drawROI(QPainter &painter);
    void drawPupil(const cv::RotatedRect ellipse, QPainter &painter);
    void drawMarker(const Marker &marker, QPainter &painter, QColor color);
    void drawGaze(const FieldData &field, QPainter &painter);
};

#endif // CAMERAWIDGET_H
