#include<thrust/for_each.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/device_vector.h>
#include<thrust/extrema.h>
#include<thrust/reduce.h>
#include<vector>
#include<iostream>
#include<fstream>
#include<stdio.h>
#include<math.h>
#include "gpu_flow_sim.h"

struct update_functor
{
	const float dt;
	const float dx;
	
	update_functor(float _dx, float _dt) : dx(_dx), dt(_dt) {}
	
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
		
		//unpack inputs from tuple
		float Qprev = thrust::get<0>(t);
		float Qcurr = thrust::get<1>(t);
		float Qnext = thrust::get<2>(t);
		float Aprev = thrust::get<3>(t);
		float Acurr = thrust::get<4>(t);
		float Anext = thrust::get<5>(t);
		float S = thrust::get<6>(t);
		float r = thrust::get<7>(t);
		
		//interpolate values at j+1/2, j-1/2
		float Qpos = (Qcurr + Qnext)/2;
		float Qneg = (Qcurr + Qprev)/2;
		float Apos = (Acurr + Anext)/2;
		float Aneg = (Acurr + Aprev)/2;
		
		
		//calculate eigenvalues
		float lambda1pos = Qpos/Apos - sqrtf(Apos);
		float lambda2pos = Qpos/Apos + sqrtf(Apos);
		float lambda1neg = Qneg/Aneg - sqrtf(Aneg);
		float lambda2neg = Qneg/Aneg + sqrtf(Aneg);
		
		float sgnlambda1pos = 1;
		if (lambda1pos < 0) sgnlambda1pos = -1;
		
		float sgnlambda2pos = 1;
		if (lambda2pos < 0) sgnlambda2pos = -1;
		
		float sgnlambda1neg = 1;
		if (lambda1neg < 0) sgnlambda1neg = -1;
		
		float sgnlambda2neg = 1;
		if (lambda2neg <0) sgnlambda2neg = -1;
		
		//entries for P matrix
		float Ppos11 =lambda1pos/sqrtf(1 + pow(lambda1pos, 2));
		float Ppos12 = lambda2pos/sqrtf(1 + pow(lambda2pos, 2));
		float Ppos21 = 1/sqrtf(1 + pow(lambda1pos, 2));
		float Ppos22 = 1/sqrtf(1 + pow(lambda2pos, 2));
		
		float Pneg11 =lambda1neg/sqrtf(1 + pow(lambda1neg, 2));
		float Pneg12 = lambda2neg/sqrtf(1 + pow(lambda2neg, 2));
		float Pneg21 = 1/sqrtf(1 + pow(lambda1neg, 2));
		float Pneg22 = 1/sqrtf(1 + pow(lambda2neg, 2));
		
		//inverse P matrix
		float Pinvpos11 = Ppos22/(Ppos11*Ppos22 - Ppos12*Ppos21);
		float Pinvpos12 = -Ppos12/(Ppos11*Ppos22 - Ppos12*Ppos21);
		float Pinvpos21 = -Ppos21/(Ppos11*Ppos22 - Ppos12*Ppos21);
		float Pinvpos22 = Ppos11/(Ppos11*Ppos22 - Ppos12*Ppos21);
		
		float Pinvneg11 = Pneg22/(Pneg11*Pneg22 - Pneg12*Pneg21);
		float Pinvneg12 = -Pneg12/(Pneg11*Pneg22 - Pneg12*Pneg21);
		float Pinvneg21 = -Pneg21/(Pneg11*Pneg22 - Pneg12*Pneg21);
		float Pinvneg22 = Pneg11/(Pneg11*Pneg22 - Pneg12*Pneg21);
		
		float Fprev1 = (Aprev*Aprev/2 + Qprev*Qprev/Aprev);
		float Fprev2 = Qprev;
		float Fcurr1 = (Acurr*Acurr/2 + Qcurr*Qcurr/Acurr);
		float Fcurr2 = Qcurr;
		float Fnext1 = (Anext*Anext/2 + Qnext*Qnext/Anext);
		float Fnext2 = Qnext;
		
		
		float Qterm2 = Ppos11*sgnlambda1pos*(Pinvpos11*(Fnext1-Fcurr1)/2 + Pinvpos12*(Fnext2-Fcurr2)/2) + Ppos12*sgnlambda2pos*(Pinvpos21*(Fnext1-Fcurr1)/2 + Pinvpos22*(Fnext2-Fcurr2)/2);
		float Aterm2 = Ppos21*sgnlambda1pos*(Pinvpos11*(Fnext1-Fcurr1)/2 + Pinvpos12*(Fnext2-Fcurr2)/2) + Ppos22*sgnlambda2pos*(Pinvpos21*(Fnext1-Fcurr1)/2 + Pinvpos22*(Fnext2-Fcurr2)/2);
		
		float Qterm4 = Pneg11*sgnlambda1neg*(Pinvneg11*(Fcurr1-Fprev1)/2 + Pinvneg12*(Fcurr2-Fprev2)/2) + Pneg12*sgnlambda2neg*(Pinvneg21*(Fcurr1-Fprev1)/2 + Pinvneg22*(Fcurr2-Fprev2)/2);
		float Aterm4 = Pneg21*sgnlambda1neg*(Pinvneg11*(Fcurr1-Fprev1)/2 + Pinvneg12*(Fcurr2-Fprev2)/2) + Pneg22*sgnlambda2neg*(Pinvneg21*(Fcurr1-Fprev1)/2 + Pinvneg22*(Fcurr2-Fprev2)/2);
		
		
		//output updated values to tuple
		thrust::get<8>(t) = Qcurr - (dt/dx)*((Fcurr1+Fnext1)/2 - Qterm2 - (Fcurr1+Fprev1)/2 + Qterm4) + Acurr*S*dt;
		thrust::get<9>(t) = Acurr - (dt/dx)*((Fcurr2+Fnext2)/2 - Aterm2 - (Fcurr2+Fprev2)/2 + Aterm4) + r*dt;
		
		
    }
};


// returns the larger of the magnitudes of the two eigenvalues at each node
struct max_eig_functor
{
	__host__ __device__
	float operator()(const float& Q, const float& A) const
	{ 
		float maglambda1 = fabs(Q/A - sqrtf(A));
		float maglambda2 = fabs(Q/A + sqrtf(A));
		
		if (maglambda1 > maglambda2) return maglambda1;
		return maglambda2;
	}
};


gpu_flow_sim::gpu_flow_sim(std::vector<float> Q0, std::vector<float> A0, std::vector<float> S, std::vector<float> r, float dx)
{
	this->Q = thrust::device_vector<float>(Q0);
	this->A = thrust::device_vector<float>(A0);
	this->S = thrust::device_vector<float>(S);
	this->r = thrust::device_vector<float>(r);
	this->dx = dx;
	this->n = S.size();
	//this->dt = 0.0001;
	this->set_dt();
	this->t = 0;
}

void gpu_flow_sim::timestep()
{
	
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(this->Q.begin(), this->Q.begin()+1, this->Q.begin()+2,    //Qprev, Qcurr, Qnext
																	this->A.begin(), this->A.begin()+1, this->A.begin()+2,  //Aprev, Acurr, Anext
																	this->S.begin()+1, this->r.begin()+1, 					//S, r
																	this->Q.begin()+1, this->A.begin()+1)),					//newQ, newA
                     thrust::make_zip_iterator(thrust::make_tuple(this->Q.end()-2, this->Q.end()-1,   this->Q.end(), 
																	this->A.end()-2, this->A.end()-1,   this->A.end(),																	
																	this->S.end()-1, this->r.end()-1,
																	this->Q.end()-1,     this->A.end()-1)),
                     update_functor(this->dx, this->dt)); // functor which implements finite volume for square channel 
	
	this->t += this->dt;
	
	return;
}

void gpu_flow_sim::output()
{
	int t_rounded_ms = int(this->t*1000 + 0.5);
	char filename[50];
	sprintf(filename, "out_%d_ms.csv", t_rounded_ms);
	std::ofstream myfile;
	myfile.open(filename);

	myfile << "X,Q,A" << std::endl;
	
    for(int i=0;i<this->n; i++)	
      {
      myfile << i*this->dx  << "," << this->Q[i] <<  "," << this->A[i] << std::endl;
      }
    myfile.close();
	return;
}

float gpu_flow_sim::set_dt()
{
	thrust::device_vector<float> maxlambda(n);
	
	thrust::transform(this->Q.begin(), this->Q.end(), this->A.begin(), maxlambda.begin(), max_eig_functor());
	
	float maxc =  *thrust::max_element(maxlambda.begin(), maxlambda.end());
	
	float cfl = 0.8;
	
	this->dt = cfl*this->dx/maxc;
	
	return this->dt;
	
}
	

