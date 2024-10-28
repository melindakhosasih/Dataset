import os    
import glob
import gzip
import argparse
import numpy as np
from tqdm import tqdm

import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

def generate_scene(args, type, n_epi):
    data_path = args.data_path.format(split=type, dataset_name=args.dataset_name)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    config = habitat.get_config(config_path=args.config_path)

    episodes = []
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset
    )
    with habitat.Env(config, dataset) as env:
        scenes = sorted(glob.glob(args.scene_paths))
        for scene in tqdm(scenes):
            if type == "train" and "20" in scene:
                continue
            elif type == "val" and ("20" not in scene or "sc2" in scene or "sc3" in scene):
                continue
            elif type == "test" and ("20" not in scene or "sc0" in scene or "sc1" in scene):
                continue
            # update scene name
            with habitat.config.read_write(env.sim.habitat_config) as env_new:
                env_new.scene = scene

            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0.0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            episode = NavigationEpisode(
                episode_id=len(episodes),
                scene_id=env.sim.habitat_config.scene,
                scene_dataset_config=env.sim.habitat_config.scene_dataset,
                start_position=env.sim.sample_navigable_point(),
                start_rotation=source_rotation,
                goals=[NavigationGoal(position=env.sim.sample_navigable_point(), radius=0.2)],
            )
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
    parser.add_argument("--data_path", type=str, default="scenes/{dataset_name}/{split}/{split}.json.gz")
    parser.add_argument("--dataset_name", type=str, default="replica_cad_baked_lighting")
    parser.add_argument("--scene_paths", type=str, default="./data/{dataset_name}/stages/*.glb")
    args = parser.parse_args()
    args.config_path = args.config_path.format(dataset_name=args.dataset_name)
    args.scene_paths = args.scene_paths.format(dataset_name=args.dataset_name)
    return args

if __name__ == "__main__":
    args = parse_args()
    for type, n_epi in zip(["train", "val", "test"], [1, 1, 1]):
        generate_scene(args, type, n_epi)