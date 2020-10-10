#include<thrust/device_vector.h>
#include<math.h>
#include<vector>
#include<iostream>
#include "gpu_flow_sim.h"

int main(void)
{
	int nsteps =  300;
	int plotevery = 100;
	int updatedtevery = 10;
	int n = 2000;
	float dx = 0.01;
	float L = n*dx;
	
	std::vector<float> Q0(n);
	std::vector<float> A0(n);
	std::vector<float> S(n);
	std::vector<float> r(n);
	
	for(int i=0; i<n; i++)
	{
		Q0[i] = 0;
		A0[i] = 1 + 0.2*exp(-5*(i*dx - L/2)*(i*dx - L/2));
		S[i] = 0;
		r[i] = 0;
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
