#include "configs.h"

PoseEstimationConfig::PoseEstimationConfig()
{

}

PoseEstimationConfig::PoseEstimationConfig(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples)
{
	this->engine_path = engine_path;
	this->max_parts = max_parts;
	this->max_objects = max_objects;
	this->peak_confidence_threshold = peak_confidence_threshold;
	this->link_threshold = link_threshold;
	this->peak_window_size = peak_window_size;
	this->line_integral_samples = line_integral_samples;
}

Resnet18Size224Config::Resnet18Size224Config(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples)
: PoseEstimationConfig(engine_path, max_parts, max_objects, peak_confidence_threshold,
		link_threshold, peak_window_size, line_integral_samples)
{
	this->num_part_types = 18;
	this->num_link_types = 21;
	this->input_size = Size(224, 224);
	this->input_nchannels = 3;
	this->output_map_size = Size(56, 56);

	// init topology
	topology.resize(this->num_link_types);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first - 1;
        topology[k][3] = skeleton[k].second - 1;
	}
}

Densenet121Size256Config::Densenet121Size256Config(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples)
: PoseEstimationConfig(engine_path, max_parts, max_objects, peak_confidence_threshold,
		link_threshold, peak_window_size, line_integral_samples)
{
	this->num_part_types = 18;
	this->num_link_types = 21;
	this->input_size = Size(256, 256);
	this->input_nchannels = 3;
	this->output_map_size = Size(64, 64);

	// init topology
	topology.resize(this->num_link_types);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first - 1;
        topology[k][3] = skeleton[k].second - 1;
	}
}

FootPose224Config::FootPose224Config(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples)
: PoseEstimationConfig(engine_path, max_parts, max_objects, peak_confidence_threshold,
		link_threshold, peak_window_size, line_integral_samples)
{
	this->num_part_types = 6;
	this->num_link_types = 6;
	this->input_size = Size(224, 224);
	this->input_nchannels = 3;
	this->output_map_size = Size(56, 56);

	// init topology
	topology.resize(this->num_link_types);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first - 1;
        topology[k][3] = skeleton[k].second - 1;
	}
}

BodyAndFeetPose224Config::BodyAndFeetPose224Config(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples,
						 int output_width, int output_height)
: PoseEstimationConfig(engine_path, max_parts, max_objects, peak_confidence_threshold,
		link_threshold, peak_window_size, line_integral_samples)
{
	this->num_part_types = 23;
	this->num_link_types = 25;
	this->input_size = Size(224, 224);
	this->input_nchannels = 3;
	this->output_map_size = Size(output_width, output_height);

	// init topology
	topology.resize(this->num_link_types);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first;
        topology[k][3] = skeleton[k].second;
	}
}

BodyOnlyPose224Config::BodyOnlyPose224Config(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples,
						 int output_width, int output_height)
: PoseEstimationConfig(engine_path, max_parts, max_objects, peak_confidence_threshold,
		link_threshold, peak_window_size, line_integral_samples)
{
	this->num_part_types = 17;
	this->num_link_types = 19;
	this->input_size = Size(224, 224);
	this->input_nchannels = 3;
	this->output_map_size = Size(output_width, output_height);

	// init topology
	topology.resize(this->num_link_types);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first;
        topology[k][3] = skeleton[k].second;
	}
}

HandPose224Config::HandPose224Config(std::string engine_path, int max_parts, int max_objects, float peak_confidence_threshold,
						 float link_threshold, int peak_window_size, int line_integral_samples)
: PoseEstimationConfig(engine_path, max_parts, max_objects, peak_confidence_threshold,
		link_threshold, peak_window_size, line_integral_samples)
{
	this->num_part_types = 21;
	this->num_link_types = skeleton.size();
	this->input_size = Size(224, 224);
	this->input_nchannels = 3;
	this->output_map_size = Size(56, 56);

	// init topology
	topology.resize(this->num_link_types);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first;
        topology[k][3] = skeleton[k].second;
	}
}
