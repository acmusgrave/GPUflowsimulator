#include<thrust/device_vector.h>
#include<math.h>
#include<vector>
#include<iostream>
#include "gpu_flow_sim.h"

int main(void)
{
	int nsteps =  1500;
	int plotevery = 500;
	int updatedtevery = 10;
	int n = 2000;
	float dx = 0.01;
	float L = n*dx;
	float sigma = 0.1;
	
	std::vector<float> Q0(n);
	std::vector<float> A0(n);
	std::vector<float> S(n);
	std::vector<float> r(n);
	
	for(int i=0; i<n; i++)
	{
		Q0[i] = 2;
		A0[i] = 1;
		S[i] = 0;
		r[i] = (1/(sigma*sqrtf(2*M_PI)))*exp(-0.5*pow((i*dx - L/2)/sigma, 2));
	}
	
	gpu_flow_sim::gpu_flow_sim simulator(Q0, A0, S, r, dx);
	
	simulator.set_dt();
	
	for (int i = 0; i<nsteps; i++)
	{
		if (i%plotevery == 0) simulator.output();
		if (i%updatedtevery == 0) simulator.set_dt();
		
		simulator.timestep();
	}
	simulator.output();
	
	return 0;
}
