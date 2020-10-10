#include<thrust/device_vector.h>
#include<math.h>
#include<vector>
#include<iostream>
#include "gpu_flow_sim.h"

int main(void)
{
	int nsteps =  3000;
	int outputevery = 1000;
	int setdtevery = 10;
	int n = 2000;
	float dx = 0.01;
	
	std::vector<float> Q0(n);
	std::vector<float> A0(n);
	std::vector<float> S(n);
	std::vector<float> r(n);
	
	for(int i=0; i<n; i++)
	{
		Q0[i] = 1;
		A0[i] = 2;
		
		if (i*dx > 5 && i*dx < 10)
		{
			S[i] = 1;
		}
		else
		{
			S[i] = 0;
		}
		
		r[i] = 0;
	}
	
	gpu_flow_sim::gpu_flow_sim simulator(Q0, A0, S, r, dx);
	
	simulator.set_dt();
	
	for (int i = 0; i<nsteps; i++)
	{
		if (i%outputevery == 0) simulator.output();
		if (i%setdtevery == 0) simulator.set_dt();
		
		simulator.timestep();
	}
	simulator.output();
	
	return 0;
}
