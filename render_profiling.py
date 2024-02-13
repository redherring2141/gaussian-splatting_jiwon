#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
#import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import sys      #JWLB_20231226
import time     #JWLB_20231226
#import nvtx     #JWLB_20240101


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, NSYSNVTX, CUDAEVENT):
    if NSYSNVTX == True:#JWLB_20240130
        pipeline.pNSYSNVTX = True#JWLB_20240131
        torch.cuda.nvtx.range_push("[JWLB-render.py-render_set]03prep_render")   #JWLB_20240101
    if CUDAEVENT == True:#JWLB_20240130
        pipeline.pCUDAEVENT = True#JWLB_20240131
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True);#JWLB_20240112        
        torch.cuda.synchronize()              #JWLB_20231226
        starter.record()#JWLB_20240112

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if NSYSNVTX == True:#JWLB_20240130
        torch.cuda.nvtx.range_pop()   #JWLB_20240101
        torch.cuda.nvtx.range_push("[JWLB-render.py-render_set]04render")   #JWLB_20240101
    if CUDAEVENT == True:#JWLB_20240130
        ender.record(); torch.cuda.synchronize(); print(f'[JWLB-render.py-render_set]03prep_render: {starter.elapsed_time(ender)}ms') #JWLB_20240112
        starter.record()#JWLB_20240112

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        #rendering = render(view, gaussians, pipeline, background, NSYSNVTX, CUDAEVENT)["render"]
        gt = view.original_image[0:3, :, :]
        #torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if NSYSNVTX == True:#JWLB_20240130
        torch.cuda.nvtx.range_pop()   #JWLB_20240101           
    if CUDAEVENT == True:#JWLB_20240130        
        ender.record(); torch.cuda.synchronize(); print(f'[JWLB-render.py-render_set]04render: {starter.elapsed_time(ender)}ms') #JWLB_20240112


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, NSYSNVTX : bool, CUDAEVENT : bool):
    with torch.no_grad():
        if NSYSNVTX == True:#JWLB_20240130
            torch.cuda.nvtx.range_push("[JWLB-render.py-render_sets]02prep_render_set")   #JWLB_20240101
        if CUDAEVENT == True:#JWLB_20240130
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True);#JWLB_20240112        
            torch.cuda.synchronize()              #JWLB_20231226
            starter.record()#JWLB_20240112

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        if NSYSNVTX == True:#JWLB_20240130
            torch.cuda.nvtx.range_pop()   #JWLB_20240101
            torch.cuda.nvtx.range_push("[JWLB-render.py-render_sets]21render_set")   #JWLB_20240101
        if CUDAEVENT == True:#JWLB_20240130
            ender.record(); torch.cuda.synchronize(); print(f'[JWLB-render.py-render_sets]02prep_render_set: {starter.elapsed_time(ender)}ms') #JWLB_20240112
            starter.record()#JWLB_20240112        
        torch.cuda.cudart().cudaProfilerStart()#JWLB_20240206
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, NSYSNVTX, CUDAEVENT)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, NSYSNVTX, CUDAEVENT)
        torch.cuda.cudart().cudaProfilerStop()#JWLB_20240206
        if NSYSNVTX == True:#JWLB_20240130
            torch.cuda.nvtx.range_pop()   #JWLB_20240130
        if CUDAEVENT == True:#JWLB_20240130            
            ender.record(); torch.cuda.synchronize(); print(f'[JWLB-render.py-render_sets]21render_set: {starter.elapsed_time(ender)}ms') #JWLB_20240112

if __name__ == "__main__":    
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--NSYSNVTX", action="store_true", default=False)  #JWLB_20240130
    parser.add_argument("--CUDAEVENT", action="store_true", default=False)   #JWLB_20240130
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    if args.NSYSNVTX == True:#JWLB_20240130
        torch.cuda.nvtx.range_push("[JWLB-render.py-main]01Initialize_system_state(RNG)")   #JWLB_20240101
    if args.CUDAEVENT == True:#JWLB_20240130
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True);#JWLB_20240112    
        torch.cuda.synchronize()  #JWLB_20231226
        starter.record()#JWLB_20240112

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.NSYSNVTX == True:#JWLB_20240130
        torch.cuda.nvtx.range_pop() #JWLB_20240101
        torch.cuda.nvtx.range_push("[JWLB-render.py-main]22render_sets")   #JWLB_20240130
    if args.CUDAEVENT == True:#JWLB_20240130
        ender.record(); torch.cuda.synchronize(); print(f'[JWLB-render.py-main]01Initialize_system_state(RNG): {starter.elapsed_time(ender)}ms') #JWLB_20240112
        starter.record()#JWLB_20240112
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.NSYSNVTX, args.CUDAEVENT)

    if args.NSYSNVTX == True:#JWLB_20240130
        torch.cuda.nvtx.range_pop() #JWLB_20240130
    if args.CUDAEVENT == True:#JWLB_20240130
        ender.record(); torch.cuda.synchronize(); print(f'[JWLB-render.py-main]22render_sets: {starter.elapsed_time(ender)}ms') #JWLB_20240112