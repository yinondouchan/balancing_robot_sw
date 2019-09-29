#include "gstreamer_video_source.h"

void GstreamerVideoSource::init(int res_width, int res_height, int framerate)
{
	video.open("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=800, height=450,"
			" framerate=60/1, format=NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx "
			"! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true", CAP_GSTREAMER);
}

void GstreamerVideoSource::read(Mat &frame)
{
	video.grab();
	video.retrieve(frame);
}
