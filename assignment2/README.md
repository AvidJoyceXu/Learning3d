# 16-825 Assignment 2
## Author: Lingyun Xu (lingyun3@andrew.cmu.edu)

- results: 
```bash
output/*
# already organized according to each part of the assignment
```

- reproduce the results:

My assignment1 code is needed for some of the visualization. Therefore, I made the homework repo public and you can clone the repo to reproduce all the results. The generated output is also attached to the git repository.

```bash
git clone https://github.com/AvidJoyceXu/Learning3d/
pwd # cd to the Learning3d directory as the root directory

bash assignment2/scripts/*.sh # Run the script for each part
```

If there is any error for scripts, you might need to create the directory hierarchy in the `output` directory, which is default to arrange as follows:

```bash
output
- 1-loss-functions
    - 1.1-voxel
    - 1.2-pc
    - 1.3-mesh
- 2-reconstructing-3d
    - eval
    - interpretation
    - mesh_0.6
    - mesh_0.8
    - mesh_1.0
    - occupancy_1.0
    - point
    - vox
```