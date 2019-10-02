/*
 * gstreamer_video_output.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef GSTREAMER_VIDEO_OUTPUT_H_
#define GSTREAMER_VIDEO_OUTPUT_H_

#include "video_output_base.h"

typedef enum {VIDEO_OUT_SHARED_MEMORY, VIDEO_OUT_SCREEN, VIDEO_OUT_TCP_SERVER} video_output_t;

class GstreamerVideoOutput : public VideoOutputBase
{
public:
	void init(int res_width, int res_height, int framerate) override;
	void init(video_output_t output, int res_width, int res_height, int framerate);
	void output_frame(Mat &frame) override;
private:
	VideoWriter writer;
};



#endif /* GSTREAMER_VIDEO_OUTPUT_H_ */
