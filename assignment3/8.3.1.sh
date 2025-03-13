CUDA_VISIBLE_DEVICES=1 python -m surface_rendering_main --config-name=volsdf_surface \
training.resume=False \
renderer.sdf_type=neus \
renderer.s=90