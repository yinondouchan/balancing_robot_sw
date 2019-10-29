#include "gstreamer_video_source.h"

#include <sstream>
#include <iostream>

void GstreamerVideoSource::init(int res_width, int res_height, int framerate)
{
	std::stringstream input_stream;
	input_stream << "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=" << res_width << ", height=" << res_height << ","
			" framerate=" << framerate << "/1, format=NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx "
			"! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true";
	video.open(input_stream.str(), CAP_GSTREAMER);
}

void GstreamerVideoSource::init(int res_width, int res_height, int argus_width, int argus_height, int framerate)
{
	std::stringstream input_stream;
	input_stream << "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=" << argus_width << ", height=" << argus_height << ","
			" framerate=" << framerate << "/1, format=NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx, width=" << res_width << ", height=" << res_height <<
			" ! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true";
	video.open(input_stream.str(), CAP_GSTREAMER);
}

void GstreamerVideoSource::read(Mat &frame)
{
	video.grab();
	video.retrieve(frame);
}
