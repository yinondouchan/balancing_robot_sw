#include "gstreamer_video_output.h"

#include <sstream>
#include <iostream>

void GstreamerVideoOutput::init(int res_width, int res_height, int framerate)
{
	init(VIDEO_OUT_SHARED_MEMORY, res_width, res_height, framerate);
}

void GstreamerVideoOutput::init(video_output_t output, int res_width, int res_height, int framerate)
{
	std::stringstream output_stream;
	switch(output)
	{
	case VIDEO_OUT_SHARED_MEMORY:
		output_stream << "appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=" << res_width
		<< ",height=" << res_height << " ! shmsink socket-path=/tmp/follow_me wait-for-connection=false";
		writer.open(output_stream.str(), CAP_GSTREAMER, 0, (double)0, cv::Size(800, 450), true);
		break;
	case VIDEO_OUT_SCREEN:
		output_stream << "appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=" << res_width
		<< ",height=" << res_height << " ! autovideosink";
		writer.open(output_stream.str(), CAP_GSTREAMER, 0, (double)0, cv::Size(800, 450), true);
		break;
	case VIDEO_OUT_TCP_SERVER:
		output_stream << "appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=(int)" << res_width
		<< ",height=(int)" << res_height << " ! jpegenc quality=85 ! tcpserversink host=0.0.0.0 port=5000";
		writer.open(output_stream.str(), CAP_GSTREAMER, 0, (double)0, cv::Size(800, 450), true);
		break;
	}
}

void GstreamerVideoOutput::output_frame(Mat &frame)
{
	writer << frame;
}
