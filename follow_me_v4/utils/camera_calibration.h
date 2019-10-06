/*
 * camera_calibration.h
 *
 *  Created on: Oct 4, 2019
 *      Author: yinon
 */

#ifndef CAMERA_CALIBRATION_H_
#define CAMERA_CALIBRATION_H_

#include "../video/video_source_base.h"
#include "../video/video_output_base.h"
#include <opencv2/imgproc.hpp>

// waveshare IMX219 camera 160 degree FOV, fisheye model calibration for 1280x720
const Matx33d CALIB_WAVESHARE_160DEG_1280x720_FISHEYE_K(603.2367723329799, 0.0, 631.040490386724,
		0.0, 606.6503954431822, 394.50828189919275,
		0.0, 0.0, 1.0);
const Vec4d CALIB_WAVESHARE_160DEG_1280x720_FISHEYE_D(-0.06211310762499229, 0.11678409244618092, -0.20084647516958823, 0.10142080878217873);

// waveshare IMX219 camera 160 degree FOV, regular model calibration for 1280x720
const Matx33d CALIB_WAVESHARE_160DEG_1280x720_K(1092.00078632749, 0, 636.0171498009227,
 	 	  0, 1059.724146734291, 357.6422405596691,
 	 	  0, 0, 1);
const Matx<float, 5, 1> CALIB_WAVESHARE_160DEG_1280x720_D(-0.6747158077043425, 0.386399736238275, 0.04168910395441938, -0.01776324754257477, 0.05548632414089306);

// waveshare IMX219 camera 160 degree FOV, fisheye model calibration for 800x450
const Matx33d CALIB_WAVESHARE_160DEG_800x450_FISHEYE_K(491.4818378531604, 0, 399.2396359075742,
 	 	  0, 493.9914347755029, 249.2580298469375,
 	 	  0, 0, 1);
const Vec4d CALIB_WAVESHARE_160DEG_800x450_FISHEYE_D(-0.0476380274048719,
 	 	0.05494749283571412,
 	 	-0.1113009871758323,
 	 	0.06567579688358684);

// waveshare IMX219 camera 160 degree FOV, regular model calibration for 800x450
const Matx33d CALIB_WAVESHARE_160DEG_800x450_K(12448.77722666533, 0, 405.427307072959,
							  0, 1648.612950720345, 247.9677442451228,
							  0, 0, 1);
const Matx<float, 5, 1> CALIB_WAVESHARE_160DEG_800x450_D(-0.9637775169599596, -82.19833770602975, 0.08865233276978146, -0.0135218006330169, 1463.385763121977);


class CameraCalibration
{
public:
	// calibrate the camera using a video source inputing images from it
	static void calibrate(VideoSourceBase &video_source, VideoOutputBase &video_output, volatile bool &sigint_received, int n_samples=50, float square_size_mm=23.5, Size chessoard_pattern = Size(6, 9));

	// undistort an image given a camera matrix and distortion coefficients
	static void undistort_image(Mat &image, Mat &undistorted_image, const Matx33d &camera_matrix, const Matx<float, 5, 1> &distortion_coefficients, double balance);
private:
};

/*
 * calibrate a fisheye camera
 */
class FisheyeCameraCalibration
{
public:
	// calibrate the camera using a video source inputing images from it
	static void calibrate(VideoSourceBase &video_source, VideoOutputBase &video_output, volatile bool &sigint_received, int n_samples=50, float square_size_mm=23.5, Size chessboard_pattern = Size(6, 9));

	// undistort an image given a camera matrix and distortion coefficients
	static void undistort_image(Mat &image, Mat &undistorted_image, const Matx33d &camera_matrix, const Vec4d &distortion_coefficients, double balance);

//	static void undistort_rect(Rect2d &rect, Rect2d &undistorted_rect)
private:
};



#endif /* CAMERA_CALIBRATION_H_ */
