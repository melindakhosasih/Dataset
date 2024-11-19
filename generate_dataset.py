import os
import argparse
import numpy as np
import logging
from typing import TYPE_CHECKING, Union, cast
from omegaconf import OmegaConf
from tqdm import tqdm

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import quat_to_coeffs
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

# [example_4]
class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass

def check_corrupted(root_dir):
    for filename in tqdm(sorted(os.listdir(root_dir))):
        if ".npz" in filename:
            try:
                path = os.path.join(root_dir, filename)
                with np.load(path) as data:
                    rgb, pos, rot = data["rgb"], data["pos"], data["rot"]
                    depth, topdown, action = data["depth"], data["topdown"], data["action"]
            except Exception as e:
                print(f"Error loading data from {path}: {e}")

def convert_to_numpy(data):
    for key in data:
        data[key] = np.array(data[key])
    return data

def generate_dataset(args, type="train", n_epi=50):
    save_dir = args.save_dir.format(split=type, dataset_name=args.dataset_name)
    log_dir = os.path.join(save_dir, "log")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # Configure logging
    logging.basicConfig(filename=os.path.join(log_dir, "warning_log.log"), level=logging.WARNING)

    # Create habitat config
    config = habitat.get_config(config_path=args.config_path)
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = type
    OmegaConf.set_readonly(config, True)

    # Add habitat.tasks.nav.nav.TopDownMap and habitat.tasks.nav.nav.Collisions measures
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    # Create dataset
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )

    # Create simulation environment
    with habitat.Env(config=config, dataset=dataset) as env:
        # Create ShortestPathFollowerAgent agent
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=config.habitat.task.measurements.success.success_distance,
        )

        for epi in (tqdm(range(env.number_of_episodes))):
            # Load the first episode and reset agent
            obs = env.reset()
            agent.reset()
            current_episode = env.current_episode

            step = 0
            scene_name, _ = os.path.splitext(os.path.basename(current_episode.scene_id))
            save_name = f"{scene_name}_{current_episode.episode_id.zfill(len(str(n_epi)))}"
            scene_info = {
                "rgb": [],
                "depth": [],
                "action": [],
                "pos": [],
                "rot": [],
                "topdown": [],
            }

            # Get metrics
            info = env.get_metrics()
            # Concatenate RGB-D observation and topdowm map into one image
            frame = observations_to_image(obs, info)

            if "top_down_map" in info:
                top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                    info["top_down_map"], obs["rgb"].shape[0]
                )

            # Remove top_down_map from metrics
            info.pop("top_down_map")
            # Overlay numeric metrics onto frame
            frame = overlay_frame(frame, info)
            # Add fame to vis_frames
            vis_frames = [frame]

            # Repeat the steps above while agent doesn't reach the goal
            while not env.episode_over:
                # Get the next best action
                action = agent.act(obs)
                if action is None:
                    break

                scene_info["rgb"].append(obs["rgb"])
                scene_info["depth"].append(obs["depth"])
                scene_info["action"].append(action)
                scene_info["pos"].append(env.sim.get_agent_state().position.tolist())
                scene_info["rot"].append(quat_to_coeffs(env.sim.get_agent_state().rotation).tolist())
                scene_info["topdown"].append(top_down_map)

                # Step in the environment
                obs = env.step(action)
                info = env.get_metrics()
                frame = observations_to_image(obs, info)

                if "top_down_map" in info:
                    top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs["rgb"].shape[0]
                    )

                # Remove top_down_map from metrics
                info.pop("top_down_map")
                # Overlay numeric metrics onto frame
                frame = overlay_frame(frame, info)
                # Add frame to vis_frames
                vis_frames.append(frame)

                step += 1
                assert step < 500, "Can't reach goal!"

            assert len(vis_frames) > 1, "Can't navigate!"

            scene_info = convert_to_numpy(scene_info)

            # Save scene info
            np.savez_compressed(
                os.path.join(save_dir, f"{save_name}.npz"),
                **scene_info
            )

            # Create video from images and save to disk
            images_to_video(
                vis_frames, save_dir, save_name, fps=6, quality=9
            )
            vis_frames.clear()
    check_corrupted(save_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/{dataset_name}.yaml")
    parser.add_argument("--dataset_name", type=str, default="replica_cad_baked_lighting")
    parser.add_argument("--save_dir", type=str, default="dataset/{dataset_name}/{split}/")
    parser.add_argument("--multigoal", type=bool, default=False)
    args = parser.parse_args()
    if args.multigoal:
        args.dataset_name = f"{args.dataset_name}_multigoal"
    args.config_path = args.config_path.format(dataset_name=args.dataset_name)
    return args

if __name__ == "__main__":
    args = parse_args()
    for type, n_epi in zip(["val", "val", "test"], [50, 50, 50]):
        generate_dataset(args, type, n_epi)