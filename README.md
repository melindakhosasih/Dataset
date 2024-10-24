# Installation

1. Create virtual environment
    ```
    conda create -n habitat python=3.9 cmake=3.14.0
    conda activate habitat
    ```

2. Install habitat-sim

    ```
    conda install habitat-sim withbullet -c conda-forge -c aihabitat
    ```

3. Download dataset
    ```
    python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset replica_cad_baked_lighting --data-path ./dataset/
    ```