/*
 * location_estimation.h
 *
 *  Created on: Sep 29, 2019
 *      Author: yinon
 */

#ifndef LOCATION_ESTIMATION_H_
#define LOCATION_ESTIMATION_H_

#include <opencv2/core.hpp>

using namespace cv;

/*
 * estimates location of an ROI in a frame
 */
class ROILocationEstimation
{
public:
	// estimate location: given an ROI and a frame return the perpendicular distance (meters) and centroid (pixels in frame)
	void estimate_location(Rect2d &roi, Mat &frame, double &distance, Point2d &centroid);
private:
};


#endif /* LOCATION_ESTIMATION_H_ */
