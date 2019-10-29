#include "tracker_base.h"

#include <opencv2/imgproc.hpp>

// initialize the tracker - set initial frame
void TrackerBase::init(Mat &frame, Rect2d ROI) {}

// update the tracker - given previous frame and ROI, find the ROI in the next frame
bool TrackerBase::update(Mat &frame, Rect2d &out_bbox) {}

// draw ROI on frame
void TrackerBase::draw_roi_on_frame(Mat& frame, Rect2d &out_bbox)
{
	draw_roi_on_frame(frame, out_bbox, Scalar(0, 255, 0));
}

// draw ROI on frame
void TrackerBase::draw_roi_on_frame(Mat& frame, Rect2d &out_bbox, Scalar color)
{
	rectangle(frame, out_bbox, color, 2, 1 );
}
