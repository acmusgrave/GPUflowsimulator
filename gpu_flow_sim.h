#ifndef GPU_FLOW_SIM_H_
#define GPU_FLOW_SIM_H_

#include<thrust/device_vector.h>
#include<vector>

class gpu_flow_sim
{
public:
	gpu_flow_sim(std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, float);
	void timestep();
	float set_dt();
	void output();
	
private:
	thrust::device_vector<float> S;
	thrust::device_vector<float> r;
	thrust::device_vector<float> Q;
	thrust::device_vector<float> A;
	int n;
	float dx;
	float dt;
	float t;
};

#endif /* GPU_FLOW_SIM_H_ */	