# RetrievalFuse

### [Paper](https://arxiv.org/pdf/2104.00024.pdf) | [Project Page](https://nihalsid.github.io/retrieval-fuse/) | [Video](https://youtu.be/HbsUU0YODqE)

> RetrievalFuse: Neural 3D Scene Reconstruction with a Database <br />
> [Yawar Siddiqui](http://niessnerlab.org/members/yawar_siddiqui/profile.html), [Justus Thies](https://justusthies.github.io/), [Fangchang Ma](https://fangchangma.github.io/), [Qi Shan](https://shanqi.github.io/), [Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html), [Angela Dai](https://www.3dunderstanding.org/index.html) <br />
> ICCV2021

<p align="center">
  <img width="100%" src="docs/teaser.jpg"/>
</p>

This repository contains the code for the ICCV 2021 paper RetrievalFuse, a novel approach for 3D reconstruction from low resolution distance field grids and from point clouds.

 In contrast to traditional generative learned models which encode the full generative process into a neural network and can struggle with maintaining local details at the scene level, we introduce a new method that directly leverages scene geometry from the training database.

### File and Folders

---

Broad code structure is as follows: 

| File / Folder | Description |
| ------------- |-------------| 
|`config/super_resolution`|Super-resolution experiment configs|
|`config/surface_reconstruction`|Surface reconstruction experiment configs|
|`config/base`|Defaults for configurations|
|`config/config_handler.py`|Config file parser|
|`data/splits`|Training and validation splits for different datasets|
|`dataset/scene.py`|SceneHandler class for managing access to scene data samples|
|`dataset/patched_scene_dataset.py`|Pytorch dataset class for scene data|
|`external/ChamferDistancePytorch`|For calculating rough chamfer distance between prediction and target while training|
|`model/attention.py`|Attention, folding and unfolding modules|
|`model/loss.py`|Loss functions|
|`model/refinement.py`|Refinement network|
|`model/retrieval.py`|Retrieval network|
|`model/unet.py`|U-Net model used as a backbone in refinement network|
|`runs/`|Checkpoint and visualizations for experiments dumped here|
|`trainer/train_retrieval.py`|Lightning module for training retrieval network|
|`trainer/train_refinement.py`|Lightning module for training refinement network|
|`util/arguments.py`|Argument parsing (additional arguments apart from those in config)|
|`util/filesystem_logger.py`|For copying source code for each run in the experiment log directory|
|`util/metrics.py`|Rough metrics for logging during training|
|`util/mesh_metrics.py`|Final metrics on meshes|
|`util/retrieval.py`|Script to dump retrievals once retrieval networks have been trained; needed for training refinement.|
|`util/visualizations.py`|Utility scripts for visualizations|

Further, the `data/` directory has the following layout


```
data                    # root data directory
├── sdf_008             # low-res (8^3) distance fields
    ├── <dataset_0>     
        ├── <sample_0>
        ├── <sample_1>
        ├── <sample_2>
        ...
    ├── <dataset_1>
    ...
├── sdf_016             # low-res (16^3) distance fields
    ├── <dataset_0>
        ├── <sample_0>
        ├── <sample_1>
        ├── <sample_2>
        ...
    ├── <dataset_1>
    ...
├── sdf_064             # high-res (64^3) distance fields
    ├── <dataset_0>
            ├── <sample_0>
            ├── <sample_1>
            ├── <sample_2>
            ...
        ├── <dataset_1>
        ...
├── pc_20K              # point cloud inputs
    ├── <dataset_0>
        ├── <sample_0>
        ├── <sample_1>
        ├── <sample_2>
        ...
    ├── <dataset_1>
    ...
├── splits              # train/val splits
├── size                # data needed by SceneHandler class (autocreated on first run)
├── occupancy           # data needed by SceneHandler class (autocreated on first run)
```
 
### Dependencies

---
Install the dependencies using pip
```bash
pip install -r requirements.txt
```
Be sure that you pull the `ChamferDistancePytorch` submodule in `external`.

### Data Preparation

---

For ShapeNetV2 and Matterport, get the appropriate meshes from the datasets. For 3DFRONT get the 3DFUTURE meshes and 3DFRONT scripts. For getting 3DFRONT meshes use [our fork of 3D-FRONT-ToolBox](https://github.com/nihalsid/3D-FRONT-ToolBox/tree/master/scripts) to create room meshes.

Once you have the meshes, use [our fork of `sdf-gen`]() to create distance field low-res inputs and high-res targets. For creating point cloud inputs simply use `trimesh.sample.sample_surface` (check `util/misc/sample_scene_point_clouds`). Place the processed data in appropriate directories:

- `data/sdf_008/<dataset>` or `data/sdf_016/<dataset>` for low-res inputs

- `data/pc_20K/<dataset>` for point clouds inputs

- `data/sdf_064/<dataset>` for targets

### Training the Retrieval Network

---

To train retrieval networks use the following command:

```bash
python trainer/train_retrieval.py --config config/<config> --val_check_interval 5 --experiment retrieval --wandb_main --sanity_steps 1
```

We provide some sample configurations for retrieval.

For super-resolution, e.g.

- `config/super_resolution/ShapeNetV2/retrieval_008_064.yaml`
- `config/super_resolution/3DFront/retrieval_008_064.yaml`
- `config/super_resolution/Matterport3D/retrieval_016_064.yaml`

For surface-reconstruction, e.g.

- `config/surface_reconstruction/ShapeNetV2/retrieval_128_064.yaml`
- `config/surface_reconstruction/3DFront/retrieval_128_064.yaml`
- `config/surface_reconstruction/Matterport3D/retrieval_128_064.yaml`

Once trained, create the retrievals for train/validation set using the following commands:
```bash
python util/retrieval.py  --mode map --retrieval_ckpt <trained_retrieval_ckpt> --config <retrieval_config>
```
```bash
python util/retrieval.py --mode compose --retrieval_ckpt <trained_retrieval_ckpt> --config <retrieval_config> 
```

### Training the Refinement Network

---

Use the following command to train the refinement network

```bash
python trainer/train_refinement.py --config <config> --val_check_interval 5 --experiment refinement --sanity_steps 1 --wandb_main --retrieval_ckpt <retrieval_ckpt>
```

Again, sample configurations for refinement are provided in the `config` directory.

For super-resolution, e.g.

- `config/super_resolution/ShapeNetV2/refinement_008_064.yaml`
- `config/super_resolution/3DFront/refinement_008_064.yaml`
- `config/super_resolution/Matterport3D/refinement_016_064.yaml`

For surface-reconstruction, e.g.

- `config/surface_reconstruction/ShapeNetV2/refinement_128_064.yaml`
- `config/surface_reconstruction/3DFront/refinement_128_064.yaml`
- `config/surface_reconstruction/Matterport3D/refinement_128_064.yaml`

### Visualizations and Logs

Visualizations and checkpoints are dumped in the `runs/<experiment>` directory. Logs are uploaded to the user's [Weights&Biases](https://wandb.ai/site) dashboard.

### Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{siddiqui2021retrievalfuse,
  title = {RetrievalFuse: Neural 3D Scene Reconstruction with a Database},
  author = {Siddiqui, Yawar and Thies, Justus and Ma, Fangchang and Shan, Qi and Nie{\ss}ner, Matthias and Dai, Angela},
  booktitle = {Proc. International Conference on Computer Vision (ICCV)},
  month = oct,
  year = {2021},
  doi = {},
  month_numeric = {10}
}
```

### License

The code from this repository is released under the MIT license.