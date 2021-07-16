
#%%
import os
import pickle as pkl 
import numpy as np
import json
import pandas as pd
import torch
import torch.multiprocessing

from bokeh.io import output_notebook, show; output_notebook()
from bokeh.plotting import gridplot

from cosypose.lib3d import Transform
from cosypose.config import EXP_DIR, MEMORY, RESULTS_DIR, LOCAL_DATA_DIR
from cosypose.datasets.datasets_cfg import make_scene_dataset, make_object_dataset
from cosypose.rendering.bullet_scene_renderer import BulletSceneRenderer
from cosypose.visualization.plotter import Plotter
from cosypose.visualization.singleview import make_singleview_prediction_plots, filter_predictions
from cosypose.visualization.singleview import filter_predictions
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
from cosypose.training.pose_models_cfg import create_model_refiner, create_model_coarse, check_update_config
from cosypose.models.efficientnet import EfficientNet
from cosypose.models.wide_resnet import WideResNet18, WideResNet34
from cosypose.models.flownet import flownet_pretrained

# Pose models
from cosypose.models.pose import PosePredictor
import yaml

#import logging
#loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
#for logger in loggers:
#   if 'cosypose' in logger.name:
#        logger.setLevel(logging.DEBUG)
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

import cosypose.utils.tensor_collection as tc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#%%
ds_name, urdf_ds_name = 'ycbv.test.keyframes', 'ycbv'

## Load dataset
scene_ds = make_scene_dataset(ds_name) # Load all frames.

## Get sample from dataset
scene_id = 48
view_id = 733

mask = scene_ds.frame_index['scene_id'] == scene_id
scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)
mask = scene_ds.frame_index['view_id'] == view_id
scene_ds.frame_index = scene_ds.frame_index[mask].reset_index(drop=True)

ds_rgb, mask, state = scene_ds[0] # Get first sample.

objects = state['objects']  # groundtruths of objects.
cameras = [state['camera']] # groundtruths of cameras.

print('The first object name and pose :')
print(objects[0]['name'])
print(objects[0]['TWO'])

#print(objects[0][0]['TWO'])

print('extrinsic and intrinsic parameters of camera')
print(cameras[0]['TWC'])
print(cameras[0]['K'])
renderer = BulletSceneRenderer(urdf_ds_name) # Create renderer.
objects_rgb = renderer.render_scene(objects, cameras)[0]['rgb'] # Render the scene using object and camera poses.
renderer.disconnect() #disconnect renderer.


## Plotting images.
plotter = Plotter()
fig_ds_rgb = plotter.plot_image(ds_rgb) 
fig_objects_rgb = plotter.plot_image(objects_rgb)
fig_overlay_gt = plotter.plot_overlay(ds_rgb, objects_rgb)

show(gridplot([[fig_ds_rgb, fig_objects_rgb, fig_overlay_gt]], sizing_mode='scale_width'))
# %%

try:
    from PIL import Image
except ImportError:
    import Image

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["animation.html"] = "jshtml"
import matplotlib.animation as animation

fig, ax = plt.subplots()

imgs = []

renderer = BulletSceneRenderer('ycbv')  # Create renderer.

for sid, sample in enumerate(scene_ds):
    input_img, _, state = sample

    camera = state['camera']
    cameras = [camera]
    objects = state['objects']

    target_object = dict(name=objects[0]['name'],
                         color='yellow',
                         TWO=objects[0]['TWO'])

    list_objects = [target_object]

    # target_objects_pose = target_objects['TWO']
    # fig_overlay = plotter.plot_overlay(input_img, rendered_img)

    rendered_img = renderer.render_scene(list_objects, cameras)[0]['rgb']

    pil_input_img = Image.fromarray(input_img.numpy())
    pil_rendered_img = Image.fromarray(rendered_img)

    pil_blend_img = Image.blend(pil_input_img, pil_rendered_img, 0.5)
    result_img = np.array(pil_blend_img)

    ax_img = ax.imshow(result_img)

    imgs.append([ax_img])

renderer.disconnect()
# ani = animation.ArtistAnimation(fig, imgs, interval=33, blit=True,repeat_delay=1000)
#%%


from cosypose.visualization.multiview import render_predictions_wrt_camera
from cosypose.scripts.run_cosypose_eval import load_posecnn_results

detections = load_posecnn_results()

# we can filter results in this way.
mask = (detections.infos['score'] >= 0.0)
detections = detections[np.where(mask)[0]]
det_index = detections.infos['scene_id'] == 48
detections = detections[np.where(det_index)]
det_index = detections.infos['view_id'] == 1
detections = detections[np.where(det_index)]

print(detections.poses[0])
print(detections.bboxes[0])

colors = ['yellow' for _ in range(len(detections.poses))]
detections.infos['color'] = colors

renderer = BulletSceneRenderer(urdf_ds_name) 
cand_rgb_rendered = render_predictions_wrt_camera(renderer, detections, state['camera'])
renderer.disconnect()

fig_detections = plotter.plot_maskrcnn_bboxes(fig_ds_rgb, detections)
fig_cand = plotter.plot_overlay(ds_rgb, cand_rgb_rendered)

show(gridplot([[fig_detections, fig_cand]], sizing_mode='scale_width'))

object_ds_name = 'ycbv.bop-compat.eval'
refiner_run_id = 'ycbv-refiner-finetune--251020'

object_ds = make_object_dataset(object_ds_name)
mesh_db = MeshDataBase.from_object_ds(object_ds)
renderer = BulletBatchRenderer(object_set=urdf_ds_name, n_workers=8)
mesh_db_batched = mesh_db.batched().cuda()


def load_model(run_id):
    if run_id is None:
        return
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / 'config.yaml').read_text(), Loader=yaml.FullLoader)
    cfg = check_update_config(cfg)

    n_inputs = 6
    backbone_str = cfg.backbone_str  # 'efficientnet-b3'
    backbone = EfficientNet.from_name(backbone_str, in_channels=n_inputs)
    backbone.n_features = 1536
    pose_dim = cfg.n_pose_dims

    # logger.info(f'Backbone: {backbone_str}')
    backbone.n_inputs = n_inputs
    render_size = (240, 320)
    model = PosePredictor(backbone=backbone,
                          renderer=renderer,
                          mesh_db=mesh_db_batched,
                          render_size=render_size,
                          pose_dim=pose_dim)

    return model

pose_predictor = load_model(refiner_run_id)
'''
for o, obj in enumerate(obs['objects']):
    obj_info = dict(
        label=obj['name'],
        score=1.0,
    )
    obj_info.update(im_info)
    bboxes.append(obj['bbox'])
    det_infos.append(obj_info)

gt_detections = tc.PandasTensorCollection(
    infos=pd.DataFrame(det_infos),
    bboxes=torch.as_tensor(np.stack(bboxes)),
)
'''
ds_rgb_tensor = torch.stack([ds_rgb])
image_tensor = ds_rgb_tensor.cuda().float().permute(0, 3, 1, 2) / 255
K = torch.as_tensor(np.stack([cameras[0]['K']])).cuda().float()

data_TCO_init = detections_ if use_detections_TCO else None
detections__ = detections_ if not use_detections_TCO else None

candidates, sv_preds = pose_predictor.get_predictions(
                image_tensor, cameras.K, detections=detections__,
                n_coarse_iterations=n_coarse_iterations,
                data_TCO_init=data_TCO_init,
                n_refiner_iterations=n_refiner_iterations,
            )


