/*
 * tracker_base.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef TRACKER_BASE_H_
#define TRACKER_BASE_H_

#include <opencv2/highgui.hpp>

using namespace cv;

/*
 * A base class for a tracker. A tracker solves the tracking problem:
 * Given a previous frame, an ROI (region of interest) in that image and the current image,
 * find the ROI in the next image. In this case the ROI is represented as a rectangular bounding box.
 */
class TrackerBase
{
public:
	// initialize the tracker - set initial frame
	virtual void init(Mat &frame, Rect2d ROI);

	// update the tracker - given previous frame and ROI, find the ROI in the next frame
	virtual bool update(Mat &frame, Rect2d &out_bbox);

	// draw ROI on frame
	void draw_roi_on_frame(Mat& frame, Rect2d &out_bbox);

	// draw ROI on frame
	void draw_roi_on_frame(Mat& frame, Rect2d &out_bbox, Scalar color);
private:
};



#endif /* TRACKER_BASE_H_ */
