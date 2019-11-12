/*
 * filters.h
 *
 *  Created on: Oct 31, 2019
 *      Author: yinon
 */

#ifndef FILTERS_H_
#define FILTERS_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <time.h>

using namespace cv;

class RectLowPassFilter
{
public:
	// default constructor
	RectLowPassFilter();

	// Constructor. As an input - time constants for each of the rectangle's fields
	RectLowPassFilter(double x_tc, double y_tc, double width_tc, double height_tc);

	// set the filter's time constants for each field
	void set_time_constants(double x_tc, double y_tc, double width_tc, double height_tc);

	// set an initial rect values to the filtered rect
	void set_initial_rect(Rect2d &rect);

	// update the filter given two rectangles: one to high-pass filter and one to low-pass filter
	Rect2d update(Rect2d &rect);
private:
	// low-pass filtered rectnagle
	Rect2d rect_lpf_internal;

	// filter time constants (seconds)
	double x_tc, y_tc, width_tc, height_tc;

	// previous timestamp
	clock_t prev_time;
};

class RectHighPassFilter
{
public:
	// default constructor
	RectHighPassFilter();

	// Constructor. As an input - time constants for each of the rectangle's fields
	RectHighPassFilter(double x_tc, double y_tc, double width_tc, double height_tc);

	// set the filter's time constants for each field
	void set_time_constants(double x_tc, double y_tc, double width_tc, double height_tc);

	// set an initial rect values to the filtered rect
	void set_initial_rect(Rect2d &rect);

	// update the filter given two rectangles: one to high-pass filter and one to low-pass filter
	Rect2d update(Rect2d &rect);
private:
	// low-pass filtered rectnagle
	Rect2d rect_hpf_internal;

	// previously obtained high pass filter rectangle
	Rect2d prev_rect_high_pass;

	// filter time constants (seconds)
	double x_tc, y_tc, width_tc, height_tc;

	// previous timestamp
	clock_t prev_time;
};

/*
 * Complementary filter for rectangles
 */
class RectComplementaryFilter
{
public:
	// default constructor
	RectComplementaryFilter();

	// Constructor. As an input - time constants for each of the rectangle's fields
	RectComplementaryFilter(double x_tc, double y_tc, double width_tc, double height_tc);

	// set the filter's time constants for each field
	void set_time_constants(double x_tc, double y_tc, double width_tc, double height_tc);

	// set an initial rect values to the filtered rect
	void set_initial_rect(Rect2d &rect);

	// update the complementary filter given two rectangles: one to high-pass filter and one to low-pass filter
	Rect2d update(Rect2d &rect_high_pass, Rect2d &rect_low_pass);
private:
	// low-pass filtered rectnagle
	Rect2d rect_lpf_internal;

	// high-pass filtered rectangle
	Rect2d rect_hpf_internal;

	// previously obtained high pass filter rectangle
	Rect2d prev_rect_high_pass;

	// filter time constants (seconds)
	double x_tc, y_tc, width_tc, height_tc;

	// previous timestamp
	clock_t prev_time;
};


#endif /* FILTERS_H_ */
