#include <opencv2/highgui.hpp>

using namespace cv;

void connect_parts(int &object_count, Mat &objects, Mat &connections, std::vector<std::array<int,4>> &topology, std::vector<int> &counts, int max_count);
