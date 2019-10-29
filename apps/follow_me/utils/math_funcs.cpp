#include "math_funcs.h"

double calculate_iou_rect2d(Rect2d rect1, Rect2d rect2)
{
	double max_left = max(rect1.x, rect2.x);
	double max_top = max(rect1.y, rect2.y);
	double min_right = min(rect1.x + rect1.width, rect2.x + rect2.width);
	double min_bottom = min(rect1.y + rect1.height, rect2.y + rect2.height);

	if ((min_right < max_left) || (min_bottom < max_top))
	{
		// rectangles are non-overlapping - IOU should be 0
		return 0;
	}

	double intersection_area = (min_right - max_left) * (min_bottom - max_top);
	double union_area = rect1.area() + rect2.area() - intersection_area;

	return intersection_area / union_area;
}
