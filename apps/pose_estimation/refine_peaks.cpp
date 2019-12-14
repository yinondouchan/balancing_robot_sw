#include "refine_peaks.hpp"
#include <iostream>


void refine_peaks(std::vector<std::vector<Point2f>> &refined_peaks, std::vector<int> &counts, std::vector<std::vector<Point2i>> &peaks, std::vector<Mat> &cmap, int window_size)
{
    int w = window_size / 2;
    int width = cmap[0].rows;
    int height = cmap[0].cols;

	for (int c = 0; c < cmap.size(); c++)
	{
		int count = counts[c];
		for (int p = 0; p < count; p++)
		{
			int i = peaks[c][p].y;
			int j = peaks[c][p].x;
			float weight_sum = 0.0f;

			for (int ii = i - w; ii < i + w + 1; ii++)
			{
				int ii_idx = ii;

				// reflect index at border
				if (ii < 0) ii_idx = -ii;
				else if (ii >= height) ii_idx = height - (ii - height) - 2;

				for (int jj = j - w; jj < j + w + 1; jj++)
				{
					int jj_idx = jj;

					// reflect index at border
					if (jj < 0) jj_idx = -jj;
					else if (jj >= width) jj_idx = width - (jj - width) - 2;

					float weight = cmap[c].at<float>(ii_idx, jj_idx);
					refined_peaks[c][p].y += weight * ii;
					refined_peaks[c][p].x += weight * jj;
					weight_sum += weight;
				}
			}

			refined_peaks[c][p].x /= weight_sum;
			refined_peaks[c][p].y /= weight_sum;
			refined_peaks[c][p].x += 0.5;
			refined_peaks[c][p].y += 0.5;
			refined_peaks[c][p].x /= height;
			refined_peaks[c][p].y /= width;
		}
	}
}
