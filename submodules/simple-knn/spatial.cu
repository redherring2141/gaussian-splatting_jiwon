/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

#include "nvToolsExt.h"	//JWLB_20240112

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
	nvtxRangePush("[JWLB-spatial.cu-distCUDA2]00_1distCUDA2");//JWLB_20240112
	cudaEvent_t start_JWLB, stop_JWLB; float msec=0; cudaEventCreate(&start_JWLB);	cudaEventCreate(&stop_JWLB);	//JWLB_20240112
	cudaEventRecord(start_JWLB);	cudaEventSynchronize(start_JWLB);	//JWLB_20240112

  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

	cudaEventRecord(stop_JWLB);		cudaEventSynchronize(stop_JWLB);	cudaEventElapsedTime(&msec, start_JWLB, stop_JWLB);	//JWLB_20240112
	std::cout << "[JWLB-spatial.cu-distCUDA2]00_1distCUDA2: " << msec << "ms" << std::endl; //JWLB_20240112
	nvtxRangePop();//JWLB_20240112

  return means;
}