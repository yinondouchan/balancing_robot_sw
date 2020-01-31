#include "paf_score_graph.hpp"

#include <iostream>

#define EPS 1e-6

void paf_score_graph(std::vector<Mat> &score_graph, std::vector<Mat> &paf, std::vector<std::array<int,4>> &topology, std::vector<int> counts, std::vector<std::vector<Point2f>> peaks, int num_integral_samples)
{
    int K = topology.size();
    int M = peaks[0].size();
    int H = paf[0].rows;
    int W = paf[0].cols;

	for (int k = 0; k < K; k++)
	{
		auto score_graph_nk = score_graph[k];
		auto paf_i_idx = topology[k][0];
		auto paf_j_idx = topology[k][1];
		auto cmap_a_idx = topology[k][2];
		auto cmap_b_idx = topology[k][3];
		auto paf_i = paf[paf_i_idx];
		auto paf_j = paf[paf_j_idx];

		auto counts_a = counts[cmap_a_idx];
		auto counts_b = counts[cmap_b_idx];
		auto peaks_a = peaks[cmap_a_idx];
		auto peaks_b = peaks[cmap_b_idx];

		for (int a = 0; a < counts_a; a++)
		{
			// compute point A
			float pa_i = peaks_a[a].y * H;
			float pa_j = peaks_a[a].x * W;

			for (int b = 0; b < counts_b; b++)
			{
				// compute point B
				float pb_i = peaks_b[b].y * H;
				float pb_j = peaks_b[b].x * W;

				// compute vector A->B
				float pab_i = pb_i - pa_i;
				float pab_j = pb_j - pa_j;

				// compute normalized vector A->B
				float pab_norm = sqrtf(pab_i * pab_i + pab_j * pab_j) + EPS;
				float uab_i = pab_i / pab_norm;
				float uab_j = pab_j / pab_norm;

				float integral = 0.0;
				float progress = 0.0;
				float increment = 1.0f / num_integral_samples;

				for (int t = 0; t < num_integral_samples; t++)
				{
					// compute integral point T
					float progress = (float) t / (float) num_integral_samples;
					float pt_i = pa_i + progress * pab_i; //(1.0 - progress) * pa_i + progress * pb_i;
					float pt_j = pa_j + progress * pab_j;//(1.0 - progress) * pa_j + progress * pb_j;

					// convert to int
					int pt_i_int = (int) pt_i;
					int pt_j_int = (int) pt_j;

					// skip point if out of bounds (will weaken integral)
					if (pt_i_int < 0) continue;
					if (pt_i_int > H) continue;
					if (pt_j_int < 0) continue;
					if (pt_j_int > W) continue;

					// get vector at integral point from PAF
					float pt_paf_i = paf_i.at<float>(pt_i_int, pt_j_int);
					float pt_paf_j = paf_j.at<float>(pt_i_int, pt_j_int);

					// compute dot product of normalized A->B with PAF vector at integral point
					float dot = pt_paf_i * uab_i + pt_paf_j * uab_j;
					integral += dot;

					progress += increment;
				}

				// normalize integral by number of samples
				integral /= num_integral_samples;
				score_graph_nk.at<float>(a, b) = integral;
			}
		}
	}
}
