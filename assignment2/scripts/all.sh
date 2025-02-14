pwd # should be in the learning3d directory with assignment1 & 2 as the sub-directory

echo "1.1. Fitting a voxel grid"
python assignment2/fit_data.py --max_iter 50000

echo "1.2. Fitting a poinc cloud"
python assignment2/fit_data.py --max_iter 10000 --type point

echo "1.3. Fitting a mesh"
python assignment2/fit_data.py --max_iter 6000 --type mesh

