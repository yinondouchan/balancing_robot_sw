#include "utils/PairGraph.hpp"
#include "utils/CoverTable.hpp"
#include "munkres.hpp"


void subMinRow(Mat &cost_graph, int nrows, int ncols)
{
  for (int i = 0; i < nrows; i++) 
  {
    // find min
    float min = cost_graph.at<float>(i, 0);
    for (int j = 0; j < ncols; j++) {
        float val = cost_graph.at<float>(i, j);
        if (val < min) {
            min = val;
        }
    }
    
    // subtract min
    for (int j = 0; j < ncols; j++) {
        cost_graph.at<float>(i, j) -= min;
    }
  }
}

void subMinCol(Mat &cost_graph, int nrows, int ncols)
{
  for (int j = 0; j < ncols; j++)
  {
    // find min
    float min = cost_graph.at<float>(0, j);
    for (int i = 0; i < nrows; i++) {
        float val = cost_graph.at<float>(i, j);
        if (val < min) {
            min = val;
        }
    }
    
    // subtract min
    for (int i = 0; i < nrows; i++) {
        cost_graph.at<float>(i, j) -= min;
    }
  }
}

void munkresStep1(Mat &cost_graph, PairGraph &star_graph, int nrows, int ncols)
{
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      if (!star_graph.isRowSet(i) && !star_graph.isColSet(j) && (cost_graph.at<float>(i, j) == 0))
      {
        star_graph.set(i, j);
      }
    }
  }
}

// returns 1 if we should exit
bool munkresStep2(const PairGraph &star_graph, CoverTable &cover_table)
{
  int k = star_graph.nrows < star_graph.ncols ? star_graph.nrows : star_graph.ncols;
  int count = 0;
  for (int j = 0; j < star_graph.ncols; j++) 
  {
    if (star_graph.isColSet(j)) 
    {
      cover_table.coverCol(j);
      count++;
    }
  }
  return count >= k;
}

bool munkresStep3(Mat &cost_graph, const PairGraph &star_graph, PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> &p, int nrows, int ncols)
{
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      if (cost_graph.at<float>(i, j) == 0 && !cover_table.isCovered(i, j))
      {
        prime_graph.set(i, j);
        if (star_graph.isRowSet(i))
        {
          cover_table.coverRow(i);
          cover_table.uncoverCol(star_graph.colForRow(i));
        }
        else
        {
          p.first = i;
          p.second = j;
          return 1;
        }
      }
    }
  }
  return 0;
}; 

void munkresStep4(PairGraph &star_graph, PairGraph &prime_graph, CoverTable &cover_table, std::pair<int, int> p)
{
  // repeat until no star found in prime's column
  while (star_graph.isColSet(p.second))
  {
    // find and reset star in prime's column 
    std::pair<int, int> s = { star_graph.rowForCol(p.second), p.second }; 
    star_graph.reset(s.first, s.second);

    // set this prime to a star
    star_graph.set(p.first, p.second);

    // repeat for prime in cleared star's row
    p = { s.first, prime_graph.colForRow(s.first) };
  }
  star_graph.set(p.first, p.second);
  cover_table.clear();
  prime_graph.clear();
}

void munkresStep5(Mat &cost_graph, const CoverTable &cover_table, int nrows, int ncols)
{
  bool valid = false;
  float min;
  for (int i = 0; i < nrows; i++)
  {
    for (int j = 0; j < ncols; j++)
    {
      if (!cover_table.isCovered(i, j))
      {
        if (!valid)
        {
          min = cost_graph.at<float>(i, j);
          valid = true;
        }
        else if (cost_graph.at<float>(i, j) < min)
        {
          min = cost_graph.at<float>(i, j);
        }
      }
    }
  }

  for (int i = 0; i < nrows; i++)
  {
    if (cover_table.isRowCovered(i))
    {
      for (int j = 0; j < ncols; j++) {
          cost_graph.at<float>(i, j) += min;
      }
//       cost_graph.addToRow(i, min);
    }
  }
  for (int j = 0; j < ncols; j++)
  {
    if (!cover_table.isColCovered(j))
    {
      for (int i = 0; i < nrows; i++) {
          cost_graph.at<float>(i, j) -= min;
      }
//       cost_graph.addToCol(j, -min);
    }
  }
}


void _munkres(Mat &cost_graph, PairGraph &star_graph, int nrows, int ncols)
{
	PairGraph prime_graph(nrows, ncols);
	CoverTable cover_table(nrows, ncols);
	prime_graph.clear();
	cover_table.clear();
	star_graph.clear();

	int step = 0;
	if (ncols >= nrows)
	{
		subMinRow(cost_graph, nrows, ncols);
	}
	if (ncols > nrows)
	{
		step = 1;
	}

	std::pair<int, int> p;
	bool done = false;
	while (!done)
	{
		switch(step)
		{
		case 0:
			subMinCol(cost_graph, nrows, ncols);
		case 1:
			munkresStep1(cost_graph, star_graph, nrows, ncols);
		case 2:
			if(munkresStep2(star_graph, cover_table))
			{
				done = true;
				break;
			}
		case 3:
			if (!munkresStep3(cost_graph, star_graph, prime_graph, cover_table, p, nrows, ncols))
			{
				step = 5;
				break;
			}
		case 4:
			munkresStep4(star_graph, prime_graph, cover_table, p);
			step = 2;
			break;
		case 5:
			munkresStep5(cost_graph, cover_table, nrows, ncols);
			step = 3;
			break;
		}
	}
}

// assignment Kx2xM
void assignment(Mat &connections, std::vector<Mat> &score_graph, std::vector<std::array<int,4>> &topology, std::vector<int> counts, float score_threshold)
{
    int K = topology.size();
    
    std::vector<Mat> cost_graph;
    for (int i = 0; i < score_graph.size(); i++)
    {
    	cost_graph.push_back(-score_graph[i]);
    }

	for (int k = 0; k < K; k++)
	{
		int cmap_a_idx = topology[k][2];
		int cmap_b_idx = topology[k][3];
		int nrows = counts[cmap_a_idx];
		int ncols = counts[cmap_b_idx];
		auto star_graph = PairGraph(nrows, ncols);
		auto cost_graph_out_a_nk = cost_graph[k];
		_munkres(cost_graph_out_a_nk, star_graph, nrows, ncols);

		auto score_graph_a_nk = score_graph[k];

		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				if (star_graph.isPair(i, j) && score_graph_a_nk.at<float>(i, j) > score_threshold) {
					connections.at<int>(k, 0, i) = j;
					connections.at<int>(k, 1, j) = i;
				}
			}
		}
	}
}
