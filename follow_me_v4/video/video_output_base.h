#ifndef VIDEO_OUTPUT_BASE_H_
#define VIDEO_OUTPUT_BASE_H_

#include <opencv2/highgui.hpp>

using namespace cv;

class VideoOutputBase
{
public:
	virtual void init(int res_width, int res_height, int framerate) = 0;
	virtual void output_frame(Mat &frame) = 0;
};

#endif // VIDEO_OUTPUT_BASE_H
