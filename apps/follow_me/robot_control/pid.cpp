#include "pid.h"

#include <iostream>

PIDController::PIDController()
{
	_p_coeff = 0;
	_i_coeff = 0;
	_d_coeff = 0;
	reset();
}

// PID controller
PIDController::PIDController(double p_coeff, double i_coeff, double d_coeff)
							: _p_coeff(p_coeff), _i_coeff(i_coeff), _d_coeff(d_coeff)
{
	reset();
}

void PIDController::set_coefficients(double p_coeff, double i_coeff, double d_coeff)
{
	_p_coeff = p_coeff;
	_i_coeff = i_coeff;
	_d_coeff = d_coeff;
}

void PIDController::reset()
{
	_i = 0;
	_prev_error_d = 0;
	_p_lpf_tc = 0;
	_i_lpf_tc = 0;
	_d_lpf_tc = 0;
	_error_lpf_p = 0;
	_error_lpf_i = 0;
	_error_lpf_d = 0;
}

// set low-pass filter time constant for the P element
void PIDController::set_p_lpf(double time_constant)
{
	_p_lpf_tc = time_constant;
}

// set low-pass filter time constant for the I element
void PIDController::set_i_lpf(double time_constant)
{
	_i_lpf_tc = time_constant;
}

// set low-pass filter time constant for the D element
void PIDController::set_d_lpf(double time_constant)
{
	_d_lpf_tc = time_constant;
}

// output the PID-controlled signal
double PIDController::control(double error)
{
	clock_t now = clock();
	double dt = (double)(now - _prev_time) / CLOCKS_PER_SEC;
	_prev_time = now;

	_error_lpf_p = _p_lpf_tc / (_p_lpf_tc + dt) * _error_lpf_p + dt / (dt + _p_lpf_tc) * error;
	_error_lpf_i = _i_lpf_tc / (_i_lpf_tc + dt) * _error_lpf_i + dt / (dt + _i_lpf_tc) * error;
	_error_lpf_d = _d_lpf_tc / (_d_lpf_tc + dt) * _error_lpf_d + dt / (dt + _d_lpf_tc) * error;

	double _p = _p_coeff * _error_lpf_p;
	_i += _i_coeff * _error_lpf_i * dt;
	_d = _d_coeff * (_error_lpf_d - _prev_error_d) / dt;

	_prev_error_d = _error_lpf_d;

	return _p + _i + _d;
}
