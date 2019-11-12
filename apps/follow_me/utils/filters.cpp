#include "filters.h"

#include <iostream>

// default constructor
RectComplementaryFilter::RectComplementaryFilter()
{
	// set initial values
	Rect2d initial(0, 0, 0, 0);
	set_initial_rect(initial);
}

// Constructor. As an input - time constants for each of the rectangle's fields
RectComplementaryFilter::RectComplementaryFilter(double x_tc, double y_tc, double width_tc, double height_tc)
{
	// set initial values
	Rect2d initial(0, 0, 0, 0);
	set_initial_rect(initial);
	set_time_constants(x_tc, y_tc, width_tc, height_tc);
}

// set the filter's time constants for each field
void RectComplementaryFilter::set_time_constants(double x_tc, double y_tc, double width_tc, double height_tc)
{
	this->x_tc = x_tc;
	this->y_tc = y_tc;
	this->width_tc = width_tc;
	this->height_tc = height_tc;
}

// set an initial rect values to the filtered rect
void RectComplementaryFilter::set_initial_rect(Rect2d &rect)
{
	rect_lpf_internal.x = rect.x;
	rect_lpf_internal.y = rect.y;
	rect_lpf_internal.width = rect.width;
	rect_lpf_internal.height = rect.height;

	rect_hpf_internal = rect_lpf_internal;
	prev_rect_high_pass = rect_hpf_internal;
}

// update the complementary filter given two rectangles: one to high-pass filter and one to low-pass filter
Rect2d RectComplementaryFilter::update(Rect2d &rect_high_pass, Rect2d &rect_low_pass)
{
	clock_t now = clock();
	double dt = (double)(now - prev_time) / CLOCKS_PER_SEC;
	prev_time = now;

	// apply low pass on low pass rectangle
	rect_lpf_internal.x = x_tc / (x_tc + dt) * rect_lpf_internal.x + dt / (dt + x_tc) * rect_low_pass.x;
	rect_lpf_internal.y = y_tc / (y_tc + dt) * rect_lpf_internal.y + dt / (dt + y_tc) * rect_low_pass.y;
	rect_lpf_internal.width = width_tc / (width_tc + dt) * rect_lpf_internal.width + dt / (dt + width_tc) * rect_low_pass.width;
	rect_lpf_internal.height = height_tc / (height_tc + dt) * rect_lpf_internal.height + dt / (dt + height_tc) * rect_low_pass.height;

	// apply high pass on high pass rectangle
	rect_hpf_internal.x = dt / (dt + x_tc) * rect_hpf_internal.x + x_tc / (x_tc + dt) * (rect_high_pass.x - prev_rect_high_pass.x);
	rect_hpf_internal.y = dt / (dt + y_tc) * rect_hpf_internal.y + y_tc / (y_tc + dt) * (rect_high_pass.y - prev_rect_high_pass.y);
	rect_hpf_internal.width = dt / (dt + width_tc) * rect_hpf_internal.width + width_tc / (width_tc + dt) * (rect_high_pass.width - prev_rect_high_pass.width);
	rect_hpf_internal.height = dt / (dt + height_tc) * rect_hpf_internal.height + height_tc / (height_tc + dt) * (rect_high_pass.height - prev_rect_high_pass.height);

	// store previous rect_heigh_pass value
	prev_rect_high_pass = rect_high_pass;

	return Rect2d(rect_lpf_internal.x + rect_hpf_internal.x, rect_lpf_internal.y + rect_hpf_internal.y,
			rect_lpf_internal.width + rect_hpf_internal.width, rect_lpf_internal.height + rect_hpf_internal.height);
}

// default constructor
RectLowPassFilter::RectLowPassFilter()
{
	// set initial values
	Rect2d initial(0, 0, 0, 0);
	set_initial_rect(initial);
}

// Constructor. As an input - time constants for each of the rectangle's fields
RectLowPassFilter::RectLowPassFilter(double x_tc, double y_tc, double width_tc, double height_tc)
{
	// set initial values
	Rect2d initial(0, 0, 0, 0);
	set_initial_rect(initial);
	set_time_constants(x_tc, y_tc, width_tc, height_tc);
}

// set the filter's time constants for each field
void RectLowPassFilter::set_time_constants(double x_tc, double y_tc, double width_tc, double height_tc)
{
	this->x_tc = x_tc;
	this->y_tc = y_tc;
	this->width_tc = width_tc;
	this->height_tc = height_tc;
}

// set an initial rect values to the filtered rect
void RectLowPassFilter::set_initial_rect(Rect2d &rect)
{
	rect_lpf_internal.x = rect.x;
	rect_lpf_internal.y = rect.y;
	rect_lpf_internal.width = rect.width;
	rect_lpf_internal.height = rect.height;
}

// update the complementary filter given two rectangles: one to high-pass filter and one to low-pass filter
Rect2d RectLowPassFilter::update(Rect2d &rect)
{
	clock_t now = clock();
	double dt = (double)(now - prev_time) / CLOCKS_PER_SEC;
	prev_time = now;

	// apply low pass on low pass rectangle
	rect_lpf_internal.x = x_tc / (x_tc + dt) * rect_lpf_internal.x + dt / (dt + x_tc) * rect.x;
	rect_lpf_internal.y = y_tc / (y_tc + dt) * rect_lpf_internal.y + dt / (dt + y_tc) * rect.y;
	rect_lpf_internal.width = width_tc / (width_tc + dt) * rect_lpf_internal.width + dt / (dt + width_tc) * rect.width;
	rect_lpf_internal.height = height_tc / (height_tc + dt) * rect_lpf_internal.height + dt / (dt + height_tc) * rect.height;

	return rect_lpf_internal;
}
