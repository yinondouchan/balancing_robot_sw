#include <opencv2/highgui.hpp>

using namespace cv;

void refine_peaks(std::vector<std::vector<Point2f>> &refined_peaks, std::vector<int> &counts, std::vector<std::vector<Point2i>> &peaks, std::vector<Mat> &cmap, int window_size);
