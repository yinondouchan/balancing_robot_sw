#include <opencv2/highgui.hpp>

using namespace cv;

void find_peaks(std::vector<int> &counts, std::vector<std::vector<Point2i>> &peaks, std::vector<Mat> &input, float threshold, int window_size, int max_count);
