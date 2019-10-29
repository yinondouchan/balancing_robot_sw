#include "camera_calibration.h"

#include <opencv2/calib3d.hpp>
#include <iostream>

void CameraCalibration::calibrate(VideoSourceBase &video_source, VideoOutputBase &video_output, volatile bool &sigint_received, int n_samples, float square_size_mm, Size chessboard_pattern)
{
	int current_nsamples = 0;

	// 3d point in real world space
	std::vector<std::vector<Point3f>> object_points;

	// 2d points in image plane.
	std::vector<std::vector<Point2f>> image_points;

	// termination criteria for the subpixel resolution algorithm
	TermCriteria subpix_termination_criteria(1, 30, 0.001);

	// input frame size
	Size frame_size;

	// start gaining samples
	while(!sigint_received && ((current_nsamples < n_samples) || n_samples == -1))
	{
		// read frame
		Mat frame;
		video_source.read(frame);

		frame_size = Size(frame.cols, frame.rows);

		Mat frame_grayscale;
		cvtColor(frame, frame_grayscale, COLOR_BGR2GRAY);

	    std::vector<Point3f> obj;
	    for (int i = 0; i < chessboard_pattern.height; i++)
	    {
	    	for (int j = 0; j < chessboard_pattern.width; j++)
	      	{
	    		obj.push_back(Point3f((float)j * square_size_mm, (float)i * square_size_mm, 0));
	      	}
	    }

		// find chess board corners in the image (3d points), return whether succeeded in finding corner
		std::vector<Point2f> corners;
		bool found_corners = findChessboardCorners(frame_grayscale, chessboard_pattern, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);

		if (found_corners)
		{
			current_nsamples++;
			std::cout << "Found corners in image. Image #" << current_nsamples << " out of " << n_samples << std::endl;

		    cornerSubPix(frame_grayscale, corners, cv::Size(5, 5), cv::Size(-1, -1),
			  		     TermCriteria(TermCriteria::Type::EPS | TermCriteria::Type::MAX_ITER, 30, 0.1));
		    drawChessboardCorners(frame, chessboard_pattern, corners, found_corners);

		    image_points.push_back(corners);
		    object_points.push_back(obj);
		}

		video_output.output_frame(frame);
	}

	// camera matrix and distortion coefficients
	Mat mtx, dist;

	// rotation and translation vectors of points
	std::vector<Mat> rvecs, tvecs;

	// obtain camera matrix, distortion coefficients and rotation and translation vectors from calibration
	std::cout << "Obtained images. Calibrating camera" << std::endl;
	calibrateCamera(object_points, image_points, frame_size, mtx, dist, rvecs, tvecs);

	std::cout << "Camera matrix: " << mtx << std::endl << std::endl;
	std::cout << "Distortion coefficients: " << dist << std::endl;
}

// undistort an image given a camera matrix and distortion coefficients
void CameraCalibration::undistort_image(Mat &image, Mat &undistorted_image, const Matx33d &camera_matrix, const Matx<float, 5, 1> &distortion_coefficients, double balance)
{
	// image dimensions of the calibration
	const Size calib_dim(image.cols, image.rows);

	Mat new_camera_matrix = getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, calib_dim, balance, calib_dim);

	Mat map1, map2;
	initUndistortRectifyMap(camera_matrix, distortion_coefficients, noArray(), new_camera_matrix, calib_dim, CV_16SC2, map1, map2);

	remap(image, undistorted_image, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}

void FisheyeCameraCalibration::calibrate(VideoSourceBase &video_source, VideoOutputBase &video_output, volatile bool &sigint_received, int n_samples, float square_size_mm, Size chessboard_pattern)
{
	int current_nsamples = 0;

	// 3d point in real world space
	std::vector<std::vector<Point3f>> object_points;

	// 2d points in image plane.
	std::vector<std::vector<Point2f>> image_points;

	// termination criteria for the subpixel resolution algorithm
	TermCriteria subpix_termination_criteria(1, 30, 0.001);

	// termination criteria for the calibration itself
	TermCriteria calib_termination_criteria(1, 30, 0.00001);

	// input frame size
	Size frame_size;

	// start gaining samples
	while(!sigint_received && ((current_nsamples < n_samples) || n_samples == -1))
	{
		// read frame
		Mat frame;
		video_source.read(frame);

		frame_size = Size(frame.cols, frame.rows);

		Mat frame_grayscale;
		cvtColor(frame, frame_grayscale, COLOR_BGR2GRAY);

	    std::vector<Point3f> obj;
	    for (int i = 0; i < chessboard_pattern.height; i++)
	    {
	    	for (int j = 0; j < chessboard_pattern.width; j++)
	      	{
	    		obj.push_back(Point3f((float)j * square_size_mm, (float)i * square_size_mm, 0));
	      	}
	    }

		// find chess board corners in the image (3d points), return whether succeeded in finding corner
		std::vector<Point2f> corners;
		bool found_corners = findChessboardCorners(frame_grayscale, chessboard_pattern, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);

		if (found_corners)
		{
			current_nsamples++;
			std::cout << "Found corners in image. Image #" << current_nsamples << " out of " << n_samples << std::endl;

		    cornerSubPix(frame_grayscale, corners, cv::Size(5, 5), cv::Size(-1, -1),
			  		     TermCriteria(TermCriteria::Type::EPS | TermCriteria::Type::MAX_ITER, 30, 0.1));
		    drawChessboardCorners(frame, chessboard_pattern, corners, found_corners);

		    image_points.push_back(corners);
		    object_points.push_back(obj);
		}

		video_output.output_frame(frame);
	}

	// intrinsics maxtix and distortion vector
	Mat mtx, dist;

	// rotation and translation vectors of points
	std::vector<Mat> rvecs, tvecs;

	// do the calibration
	std::cout << "Obtained images. Calibrating camera" << std::endl;
	int calib_flags = fisheye::CALIB_RECOMPUTE_EXTRINSIC + fisheye::CALIB_CHECK_COND + fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_points, image_points, frame_size, mtx, dist, rvecs, tvecs, calib_flags, calib_termination_criteria);

	std::cout << "Camera matrix: " << mtx << std::endl << std::endl;
	std::cout << "Distortion coefficients: " << dist << std::endl;
}

// undistort an image given a camera matrix and distortion coefficients
void FisheyeCameraCalibration::undistort_image(Mat &image, Mat &undistorted_image, const Matx33d &camera_matrix, const Vec4d &distortion_coefficients, double balance)
{
	// image dimensions of the calibration
	const Size calib_dim(image.cols, image.rows);

	Matx33d new_camera_matrix = camera_matrix;
	fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix, distortion_coefficients, calib_dim, noArray(), new_camera_matrix, balance);

	Mat map1, map2;
	fisheye::initUndistortRectifyMap(camera_matrix, distortion_coefficients, noArray(), new_camera_matrix, calib_dim, CV_16SC2, map1, map2);

	remap(image, undistorted_image, map1, map2, INTER_LINEAR, BORDER_CONSTANT);
}
