import os    
import random
import gzip
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import habitat
import pointnav_generator

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

def generate_pointnav(args, type, n_epi, seed):
    data_path = args.data_path.format(split=type, dataset_name=args.dataset_name)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    config = habitat.get_config(config_path=args.config_path)
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = type
    OmegaConf.set_readonly(config, True)

    episodes = []
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    with habitat.Env(config, dataset) as env:
        env.seed(seed)
        random.seed(seed)

        for _ in (tqdm(range(env.number_of_episodes))):
            # go to next scene
            env.reset()
            # generate point nav episode
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
    parser.add_argument("--config_path", type=str, default="./config/{dataset_name}.yaml")
    parser.add_argument("--data_path", type=str, default="pointnav/{dataset_name}/{split}/{split}.json.gz")
    parser.add_argument("--dataset_name", type=str, default="replica_cad_baked_lighting")
    args = parser.parse_args()
    args.config_path = args.config_path.format(dataset_name=args.dataset_name)
    return args

if __name__ == "__main__":
    args = parse_args()
    for type, n_epi, seed in zip(["train", "val", "test"], [50, 5, 5], [500, 1000, 1500]):
        generate_pointnav(args, type, n_epi, seed)