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
from gaussian_renderer import render_orig_3dgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model_3dgs import GaussianModel
import time
import json

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # add fps calculation
    ellipse_time = 0.0
    times = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # add fps calculation
        torch.cuda.synchronize()
        tic = time.time()

        rendering = render_orig_3dgs(view, gaussians, pipeline, background)["render"]

        torch.cuda.synchronize()
        elapsed = max(time.time() - tic, 1e-10)
        ellipse_time += elapsed
        times.append(elapsed)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    if len(times) > 3:
        avg_time = sum(times[3:]) / (len(times) - 3)
    elif times:
        avg_time = sum(times) / len(times)
    else:
        avg_time = 0.0
    ellipse_time = avg_time
    time_per_image = avg_time
    fps = 1.0 / max(avg_time, 1e-10)

    # print to stdout
    print("times " + str(times))
    print(f"fps {fps}")

    # save fps and times to JSON under render_path
    fps_json_path = os.path.join(model_path, "fps.json")
    fps_payload = {
        "fps": fps,
        "time_per_image": time_per_image,
        "times": times,
        "num_images": len(times),
    }
    try:
        with open(fps_json_path, "w") as f:
            json.dump(fps_payload, f, indent=2)
    except Exception as e:
        print(f"Failed to write fps.json to {fps_json_path}: {e}")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)