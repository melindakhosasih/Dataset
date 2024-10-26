import os    
from tqdm import tqdm
import random
import glob
import gzip

import habitat
import pointnav_generator

CFG_TEST = "./config/pointnav.yaml"
NUM_EPISODES = 20

if __name__ == "__main__":
    config = habitat.get_config(config_path=CFG_TEST)
    
    episodes = []
    with habitat.Env(config) as env:
        env.seed(config.habitat.seed)
        random.seed(config.habitat.seed)

        scenes = sorted(glob.glob("./data/replica_cad_baked_lighting/stages/*.glb"))
        for scene in tqdm(scenes):
            # update scene name
            with habitat.config.read_write(env.sim.habitat_config) as env_new:
                env_new.scene = scene

            generator = pointnav_generator.generate_pointnav_episode(
                sim=env.sim,
                num_episodes=NUM_EPISODES,
                shortest_path_success_distance=config.habitat.task.measurements.success.success_distance,
                shortest_path_max_steps=config.habitat.environment.max_episode_steps,
                closest_dist_limit=10,
            )
            for episode in generator:
                episodes.append(episode)
            
    dataset: habitat.Dataset = habitat.Dataset()
    dataset.episodes = episodes
    assert (
        dataset.to_json()
    ), "Generated episodes aren't json serializable."
    out = dataset.to_json()
    root_dir = "pointnav"
    os.makedirs(root_dir, exist_ok=True)
    with gzip.open(os.path.join(root_dir, "replica_cad_baked.json.gz"), "wt", encoding="utf-8") as file:
        file.write(out)