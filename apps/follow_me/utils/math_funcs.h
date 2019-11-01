/*
 * math_funcs.h
 *
 *  Created on: Oct 28, 2019
 *      Author: yinon
 */

#ifndef MATH_FUNCS_H_
#define MATH_FUNCS_H_

#include <opencv2/highgui.hpp>

using namespace cv;

// calculate the intersection-over-union between two rectangles
double calculate_iou_rect2d(Rect2d rect1, Rect2d rect2);


#endif /* MATH_FUNCS_H_ */
