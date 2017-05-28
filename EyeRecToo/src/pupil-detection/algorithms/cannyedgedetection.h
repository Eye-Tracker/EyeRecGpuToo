#ifndef CANNY_EDGE_DETECTION_H
#define CANNY_EDGE_DETECTION_H

#include <opencv2/imgproc/imgproc.hpp>
#include "cannyedgedetectioncpu.h"
#include "cannyedgedetectiongpu.h"

class Algorithm;

class CannyEdgeDetectorSelector {
public:
    enum Type {
        CPU, GPU
    };

    CannyEdgeDetectorSelector() {
        algorithm_ = nullptr;
    }

    void setAlgorithmType(int type, cv::Mat *pic);
    cv::Mat execute();

private:
    Algorithm *algorithm_;
};

class Algorithm {
public:
    Algorithm(cv::Mat *pic) {
        pic_ = pic;
    }

    virtual cv::Mat execute();

protected:
    cv::Mat *pic_;
};

void CannyEdgeDetectorSelector::setAlgorithmType(int type, cv::Mat *pic) {
    delete algorithm_;
    if (type == CPU)
        algorithm_ = new CannyEdgeDetectionCPU(pic);
    else if (type == GPU)
        algorithm_ = new CannyEdgeDetectionGPU(pic);
}

cv::Mat CannyEdgeDetectorSelector::execute() {
    return algorithm_->execute();
}

#endif // CANNY_EDGE_DETECTION_H
