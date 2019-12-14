#include "find_peaks.hpp"

/*
 * Find peaks in confidence maps.
 *
 * Inputs:
 *
 * input - vector of confidence maps for each joint type
 * threshold - confidence threshold for outputting a peak
 * window_size - window size for checking whether a point is a peak
 * max_count - maximum joint count for one joint type
 *
 * Outputs:
 * counts - vector of peak counts for each joint type
 * peaks - vector of vector of peak points for each joint type
 */
void find_peaks(std::vector<int> &counts, std::vector<std::vector<Point2i>> &peaks, std::vector<Mat> &input, float threshold, int window_size, int max_count)
{
    int w = window_size / 2;
    int width = input[0].cols;
    int height = input[0].rows;

	for (int c = 0; c < input.size(); c++)
	{
		int count = 0;

		for (int i = 0; i < height && count < max_count; i++)
		{
			for (int j = 0; j < width && count < max_count; j++)
			{
				float value = input[c].at<float>(i, j);

				if (value < threshold)
					continue;

				int ii_min = i - w;
				int jj_min = j - w;
				int ii_max = i + w + 1;
				int jj_max = j + w + 1;

				if (ii_min < 0) ii_min = 0;
				if (ii_max > height) ii_max = height;
				if (jj_min < 0) jj_min = 0;
				if (jj_max > width) jj_max = width;

				// get max
				bool is_peak = true;
				for (int ii = ii_min; ii < ii_max; ii++)
				{
					for (int jj = jj_min; jj < jj_max; jj++)
					{
						if (input[c].at<float>(ii, jj) > value) {
							is_peak = false;
						}
					}
				}

				if (is_peak) {
					peaks[c][count].y = i;
					peaks[c][count].x = j;
					count++;
				}
			}
		}

		counts[c] = count;
	}
}
