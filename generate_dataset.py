# [setup]
import os
import argparse
import numpy as np
import logging
from omegaconf import OmegaConf
from tqdm import tqdm

import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
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
        for epi in (tqdm(range(env.number_of_episodes))):
            _ = env.reset()
            current_episode = env.current_episode
            assert epi % n_epi == int(current_episode.episode_id), (
                f"Episode {epi % n_epi} doesn't match with current epi {current_episode.episode_id}"
            )

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
            vis_frames = []
            obs = env.sim.get_observations_at(current_episode.start_position, current_episode.start_rotation, True)
            for step, path in enumerate(current_episode.shortest_paths[0]):
                if path.position != env.sim.get_agent_state().position.tolist():
                    logging.warning(
                        f"{save_name}, at step {step} position is not equal! "
                        f"Dataset pos: {path.position}, "
                        f"Agent pos: {env.sim.get_agent_state().position.tolist()}"
                    )
                if path.rotation != quat_to_coeffs(env.sim.get_agent_state().rotation).tolist():
                    logging.warning(
                        f"{save_name}, at step {step} rotation is not equal! "
                        f"Dataset rot: {path.rotation}, "
                        f"Agent rot: {quat_to_coeffs(env.sim.get_agent_state().rotation).tolist()}"
                    )
                # Get action from dataset
                action = path.action

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
                # Add frame to vis_frames
                vis_frames.append(frame)

                # Store info
                scene_info["rgb"].append(obs["rgb"])
                scene_info["depth"].append(obs["depth"])
                scene_info["action"].append(action)
                scene_info["pos"].append(env.sim.get_agent_state().position.tolist())
                scene_info["rot"].append(quat_to_coeffs(env.sim.get_agent_state().rotation).tolist())
                scene_info["topdown"].append(top_down_map)

                obs = env.step(action)
                assert step < 500, "Can't reach goal!"
            
            # Store last step info
            scene_info["rgb"].append(obs["rgb"])
            scene_info["depth"].append(obs["depth"])
            scene_info["action"].append(0)
            scene_info["pos"].append(env.sim.get_agent_state().position.tolist())
            scene_info["rot"].append(quat_to_coeffs(env.sim.get_agent_state().rotation).tolist())
            scene_info["topdown"].append(top_down_map)

            scene_info = convert_to_numpy(scene_info)

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
    args = parser.parse_args()
    args.config_path = args.config_path.format(dataset_name=args.dataset_name)
    return args

if __name__ == "__main__":
    args = parse_args()
    for type, n_epi in zip(["train", "val", "test"], [50, 5, 5]):
        generate_dataset(args, type, n_epi)