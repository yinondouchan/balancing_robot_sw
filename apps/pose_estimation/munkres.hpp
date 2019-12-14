#include <opencv2/highgui.hpp>

#include "utils/PairGraph.hpp"

using namespace cv;


void _munkres(Mat &cost_graph, PairGraph &star_graph, int nrows, int ncols);

// assignment NxKx2xM
void assignment(Mat &connections, std::vector<Mat> &score_graph, std::vector<std::array<int,4>> & topology, std::vector<int> counts, float score_threshold);
