#ifndef ELSE_H
#define ELSE_H

/*
  Version 1.0, 17.12.2015, Copyright University of Tübingen.

  The Code is created based on the method from the paper:
  "ElSe: Ellipse Selection for Robust Pupil Detection in Real-World Environments", W. Fuhl, T. C. Santini, T. C. Kübler, E. Kasneci
  ETRA 2016 : Eye Tracking Research and Application 2016

  The code and the algorithm are for non-comercial use only.
*/

#include <opencv2/imgproc/imgproc.hpp>

#include "PupilDetectionMethod.h"

class ElSe : public PupilDetectionMethod
{
public:
    ElSe() { mDesc = desc; }
    cv::RotatedRect run(const cv::Mat &frame);
    bool hasPupilOutline() { return true; }
    static std::string desc;
};

#endif // ELSE_H
