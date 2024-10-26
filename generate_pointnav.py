import os    
from tqdm import tqdm
import random
import glob
import gzip
import argparse

import habitat
import pointnav_generator

def generate_pointnav(args):
    config = habitat.get_config(config_path=args.config_path)
    for type, n_epi, seed in zip(["train", "val", "test"], [50, 5, 5], [500, 1000, 1500]):
        data_path = args.data_path.format(split=type, dataset_name=args.dataset_name)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        episodes = []
        with habitat.Env(config) as env:
            env.seed(seed)
            random.seed(seed)

            scenes = sorted(glob.glob(args.scene_paths))
            for scene in tqdm(scenes):
                # update scene name
                with habitat.config.read_write(env.sim.habitat_config) as env_new:
                    env_new.scene = scene

                generator = pointnav_generator.generate_pointnav_episode(
                    sim=env.sim,
                    num_episodes=n_epi,
                    shortest_path_success_distance=config.habitat.task.measurements.success.success_distance,
                    shortest_path_max_steps=config.habitat.environment.max_episode_steps,
                    closest_dist_limit=5,
                )
                for episode in generator:
                    episodes.append(episode)

        dataset: habitat.Dataset = habitat.Dataset()
        dataset.episodes = episodes
        assert (
            dataset.to_json()
        ), "Generated episodes aren't json serializable."
        out = dataset.to_json()
        with gzip.open(data_path, "wt", encoding="utf-8") as file:
            file.write(out)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/pointnav.yaml")
    parser.add_argument("--data_path", type=str, default="pointnav/{dataset_name}/{split}/{split}.json.gz")
    parser.add_argument("--dataset_name", type=str, default="replica_cad_baked_lighting")
    parser.add_argument("--scene_paths", type=str, default="./data/{dataset_name}/stages/*.glb")
    args = parser.parse_args()
    args.scene_paths = args.scene_paths.format(dataset_name=args.dataset_name)
    return args

if __name__ == "__main__":
    args = parse_args()
    generate_pointnav(args)