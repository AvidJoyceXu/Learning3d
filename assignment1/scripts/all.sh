python scripts/create_dir.py

echo "Rendering 1.1 Panaroma"
python starter/panaroma_render.py --mesh_path data/cow.obj

echo "Rendering 1.2 Dolly Zoom"
python -m starter.dolly_zoom

echo "Rendering 2.1 Tetrahedron"
python starter/panaroma_render.py --mesh_path data/MyMeshes/tetrahedron.obj --output_path play/2-practicing-meshes/2.1-tetrahedron.gif

echo "Rendering 2.2 Cube"
python starter/panaroma_render.py --mesh_path data/MyMeshes/cube.obj --output_path play/2-practicing-meshes/2.2-cube.gif

echo "Rendering 3 Retexturing"
python starter/retexture_mesh.py

echo "Rendering 4 Camera Transformation"
python starter/camera_transforms.py

echo "Rendering 5.1 Pointcloud from RGBD"
python starter/render_pc.py

echo "Rendering 5.2 Parametric Functions"
python -m starter.render_generic --render torus --output_path play/5-rendering-pc/5.2-parametric-functions/torus.gif
python -m starter.render_generic --render bottle --output_path play/5-rendering-pc/5.2-parametric-functions/bottle.gif

echo "Rendering 5.3 Implicit Surfaces"
python -m starter.render_generic --render torus_mesh --output_path play/5-rendering-pc/5.3-implicit-surfaces/torus_mesh.gif
python starter/render_generic.py --render heart_mesh --output_path play/5-rendering-pc/5.3-implicit-surfaces/pink_heart.gif

echo "Rendering 6 Fun"
python starter/fun_stuff.py

echo "Rendering 7 Sample Points on Meshes"
python starter/sample_mesh.py
