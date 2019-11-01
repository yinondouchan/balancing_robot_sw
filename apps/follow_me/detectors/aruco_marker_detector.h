/*
 * aruco_detector.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef ARUCO_DETECTOR_H_
#define ARUCO_DETECTOR_H_

#include "detector_base.h"

#include <opencv2/aruco.hpp>

class ArucoMarkerDetector : public DetectorBase
{
public:
    void init() override;

    void detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names) override;
    void detect(Mat &image, std::vector<std::vector<Point2f>> &out_polygons, std::vector<std::string> &class_names) override;
private:
    Ptr<aruco::Dictionary> _dictionary;
    Ptr<aruco::DetectorParameters> _detector_params;
};


#endif /* ARUCO_DETECTOR_H_ */
