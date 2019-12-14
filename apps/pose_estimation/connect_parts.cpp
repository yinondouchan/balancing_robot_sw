#include "connect_parts.hpp"

#include <cstdint>
#include <queue>


void connect_parts(int &object_count, Mat &objects, Mat &connections, std::vector<std::array<int,4>> &topology, std::vector<int> &counts, int max_count)
{
    int K = topology.size();
    int C = counts.size();
    int M = connections.size[2];

    Mat visited = Mat::zeros(Size(C, M), CV_8SC1);

	int num_objects = 0;
	for (int c = 0; c < C; c++)
	{
		if (num_objects >= max_count) {
			break;
		}

		int count = counts[c];

		for (int i = 0; i < count; i++)
		{
			if (num_objects >= max_count) {
				break;
			}

			std::queue<std::pair<int, int>> q;
			bool new_object = false;
			q.push({c, i});

			while (!q.empty())
			{
				auto node = q.front();
				q.pop();
				int c_n = node.first;
				int i_n = node.second;

				if (visited.at<int8_t>(c_n, i_n)) {
					continue;
				}

				visited.at<int8_t>(c_n, i_n) = 1;
				new_object = true;
				objects.at<int>(num_objects, c_n) = i_n;

				for (int k = 0; k < K; k++)
				{
					int c_a = topology[k][2];
					int c_b = topology[k][3];

					if (c_a == c_n)
					{
						int i_b = connections.at<int8_t>(k, 0, i_n);
						if (i_b >= 0) {
							q.push({c_b, i_b});
						}
					}

					if (c_b == c_n)
					{
						int i_a = connections.at<int8_t>(k, 1, i_n);
						if (i_a >= 0) {
							q.push({c_a, i_a});
						}
					}
				}
			}

			if (new_object)
			{
				num_objects++;
			}
		}
	}

	object_count = num_objects;
}
