/*
 * paf_score_graph.hpp
 *
 *  Created on: Dec 13, 2019
 *      Author: yinon
 */

#ifndef PAF_SCORE_GRAPH_HPP_
#define PAF_SCORE_GRAPH_HPP_

#include <opencv2/highgui.hpp>

using namespace cv;

void paf_score_graph(std::vector<Mat> &score_graph, std::vector<Mat> &paf, std::vector<std::array<int,4>> &topology, std::vector<int> counts, std::vector<std::vector<Point2f>> peaks, int num_integral_samples);

#endif /* PAF_SCORE_GRAPH_HPP_ */
