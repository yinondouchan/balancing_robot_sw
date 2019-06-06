class PIDController
{	
	public:
		PIDController(double p_coeff, double i_coeff, double d_coeff);
		
		double control(double error);
		
		// enable low pass filter on p, i, and d components
		void set_p_lpf(double time_constant);
		void set_i_lpf(double time_constant);
		void set_d_lpf(double time_constant);
		
	private:
		double _p_coeff, _i_coeff, _d_coeff;
		double _i;
		double _d;
		double _prev_error_d;
		
		double _error_lpf_p, _error_lpf_i, _error_lpf_d;
		double _p_lpf_tc, _i_lpf_tc, _d_lpf_tc;
		
		clock_t _prev_time;
};
