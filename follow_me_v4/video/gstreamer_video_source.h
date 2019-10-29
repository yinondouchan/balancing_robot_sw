/*
 * gstreamer.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef GSTREAMER_VIDEO_SOURCE_H_
#define GSTREAMER_VIDEO_SOURCE_H_

#include <opencv2/highgui.hpp>

#include "video_source_base.h"

using namespace cv;

class GstreamerVideoSource : public VideoSourceBase
{
public:
	void init(int res_width, int res_height, int framerate) override;
	void init(int res_width, int res_height, int argus_width, int argus_height, int framerate);
	void read(Mat &frame) override;
private:
	VideoCapture video;
};


#endif /* GSTREAMER_VIDEO_SOURCE_H_ */
