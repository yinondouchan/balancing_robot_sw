/*
 * pid.h
 *
 *  Created on: Sep 28, 2019
 *      Author: yinon
 */

#ifndef PID_H_
#define PID_H_

#include <time.h>

/*
 * An implementation of a PID controller with an option of low-pass filtering the error signal
 */
class PIDController
{
	public:
		// default constructor. p, i and d coeffitiens are not initialized - must call set_coefficients to set those
		PIDController();

		// constructor with setting the p, i and d coefficients
		PIDController(double p_coeff, double i_coeff, double d_coeff);

		// set the P, I and D coefficients
		void set_coefficients(double p_coeff, double i_coeff, double d_coeff);

		// output the PID-controlled signal
		double control(double error);

		// enable low pass filter on p, i, and d components
		void set_p_lpf(double time_constant);
		void set_i_lpf(double time_constant);
		void set_d_lpf(double time_constant);

		// reset the PID controller
		void reset();

	private:
		// P, I and D coefficients
		double _p_coeff, _i_coeff, _d_coeff;

		// integral element
		double _i;

		// derivative element
		double _d;

		// previous error (for calculating the D element)
		double _prev_error_d;

		// low-pass filters for P, I and D elements
		double _error_lpf_p, _error_lpf_i, _error_lpf_d;

		// P, I and D low pass filter time constants
		double _p_lpf_tc, _i_lpf_tc, _d_lpf_tc;

		// previous timestamp
		clock_t _prev_time;
};


#endif /* PID_H_ */
