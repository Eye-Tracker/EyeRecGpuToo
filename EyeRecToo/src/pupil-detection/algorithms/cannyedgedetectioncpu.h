#ifndef CANNYEDGEDETECTIONCPU_H
#define CANNYEDGEDETECTIONCPU_H

#include "cannyedgedetection.h"

class CannyEdgeDetectionCPU: public Algorithm {
public:
    CannyEdgeDetectionCPU(cv::Mat *pic): Algorithm(pic){}
    cv::Mat execute();
private:
    void edgetrace(cv::Mat *strong, cv::Mat *weak, cv::Mat *check);
};

#endif // CANNYEDGEDETECTIONCPU_H
