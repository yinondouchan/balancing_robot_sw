#include <opencv2/highgui.hpp>

using namespace cv;

void find_peaks(std::vector<int> &counts, std::vector<std::vector<Point2i>> &peaks, std::vector<Mat> &input, float threshold, int window_size, int max_count);
void find_peaks_optimized(std::vector<int> &counts, std::vector<std::vector<Point2i>> &peaks, const std::vector<Mat> &input, const float threshold, const int window_size, const int max_count);
