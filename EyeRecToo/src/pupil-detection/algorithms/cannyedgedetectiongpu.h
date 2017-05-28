#ifndef CANNYEDGEDETECTIONGPU_H
#define CANNYEDGEDETECTIONGPU_H

#include "cannyedgedetection.h"

class CannyEdgeDetectionGPU {
public:
    CannyEdgeDetectionGPU(cv::Mat *pic): Algorithm(pic){}
    cv::Mat execute();
};

#endif // CANNYEDGEDETECTIONGPU_H
