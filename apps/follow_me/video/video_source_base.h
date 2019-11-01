#ifndef VIDEO_SOURCE_BASE_H
#define VIDEO_SOURCE_BASE_H

#include <opencv2/highgui.hpp>

using namespace cv;

class VideoSourceBase
{
public:
	virtual void init(int res_width, int res_height, int framerate) = 0;
	virtual void read(Mat &frame) = 0;
private:
};

#endif // VIDEO_SOURCE_BASE_H
