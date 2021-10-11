import os
from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from trimesh import sample

from util.visualization import visualize_pointcloud, visualize_sdf_as_mesh


def to_point_list(s):
    return np.concatenate([c[:, np.newaxis] for c in np.where(s)], axis=1)


def read_list(path):
    return [x.strip() for x in Path(path).read_text().split("\n") if x.strip() != ""]


def rename_state_dict(state_dict, key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(key):
            new_state_dict[".".join(k.split(".")[1:])] = state_dict[k]
    return new_state_dict


def load_net_and_set_to_eval(net, ckpt_path, rename_prefix):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    net.load_state_dict(rename_state_dict(ckpt['state_dict'], rename_prefix))
    net = net.cuda()
    net.eval()
    return net


def array3d_to_tensor5d(arr):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def tensor3d_to_tensor5d(_tensor):
    return _tensor.unsqueeze(0).unsqueeze(0)


def tensor5d_to_array3d(_tensor):
    return _tensor.detach().squeeze(0).squeeze(0).cpu().numpy()


def get_iou_matrix(batch_shapes):
    #  batch_shapes: N x 1 x D x H x W
    n, _, d, h, w = batch_shapes.shape
    lhs = batch_shapes.bool().expand(-1, n, -1, -1, -1).reshape((n * n, 1, d, h, w))
    rhs = batch_shapes.bool().reshape((1, n, d, h, w)).expand(n, -1, -1, -1, -1).reshape((n * n, 1, d, h, w))
    intersection = (lhs & rhs).squeeze(1).sum(-1).sum(-1).sum(-1)
    union = (lhs | rhs).squeeze(1).sum(-1).sum(-1).sum(-1)
    iou = (intersection / (union + 1e-5)).reshape((n, n))
    return iou


def get_retrievals_dir(config):
    ckpt_experiment = Path(config['retrieval_ckpt']).parents[0].name
    ckpt_epoch = Path(config['retrieval_ckpt']).name.split('.')[0]
    num_points = config['dataset_train']['num_points']
    task_dir = f"{config['task']}_{num_points:04d}"
    # if config['task'] == 'surface_reconstruction' and not Path(config['dataset_train']['retrieval_dir'], 'retrieval', task_dir, config['dataset_train']['dataset_name'], config['dataset_train']['splits_dir']).exists():
    #     task_dir = f"{config['task']}_1000"
    print(f'Using taskdir: {task_dir}')
    return Path(config['dataset_train']['retrieval_dir'], 'retrieval', task_dir, config['dataset_train']['dataset_name'], config['dataset_train']['splits_dir'], ckpt_experiment, ckpt_epoch, str(config['K']))


def point_cloud_to_grid(point_cloud, grid_res, scale_factor, pad):
    grid = np.zeros([grid_res + 2 * pad] * 3, dtype=np.float32)
    point_cloud = point_cloud * scale_factor
    points_grid = np.clip(point_cloud, 0, grid_res - 1).astype(np.uint32)
    grid[pad + points_grid[:, 0], pad + points_grid[:, 1], pad + points_grid[:, 2]] = 1
    return grid


def create_combined_point_clouds(config, visualize):
    num_points = [2000, 1000, 500]
    split_shapes = read_list(f"{config['dataset_train']['data_dir']}/splits/{config['dataset_train']['dataset_name']}/{config['dataset_train']['splits_dir']}/train.txt")
    split_shapes.extend(read_list(f"{config['dataset_train']['data_dir']}/splits/{config['dataset_train']['dataset_name']}/{config['dataset_train']['splits_dir']}/val.txt"))
    all_point_clouds = list(Path(f"{config['dataset_train']['data_dir']}/{config['dataset_train']['input_dir']}/{config['dataset_train']['dataset_name']}").iterdir())
    all_scenes = ["__".join(s.split('__')[:2]) for s in split_shapes]
    for scene in tqdm(all_scenes):
        # find all the point clouds in the directory for that scene
        scene_point_clouds = {k: [] for k in num_points}
        for p in all_point_clouds:
            if p.name.split('.npz')[0].startswith(scene):
                point_cloud = np.load(str(p))['arr_0']
                for n in num_points:
                    rand_indices = random.sample(range(20000), n)
                    subpoint_cloud = point_cloud[rand_indices, :]
                    shift = [int(x) for x in p.name.split('.npz')[0].split('__')[-1].split('_')]
                    subpoint_cloud[:, 0] += shift[0]
                    subpoint_cloud[:, 1] += shift[1]
                    subpoint_cloud[:, 2] += shift[2]
                    scene_point_clouds[n].append(subpoint_cloud)

        for n in num_points:
            output_dir = Path(config['dataset_train']['data_dir']) / config['dataset_train']['dataset_name'] / f"pc_{n}"
            output_dir.mkdir(exist_ok=True, parents=True)
            if len(scene_point_clouds[n]) > 0:
                pc = np.vstack(scene_point_clouds[n])
                np.savez_compressed(output_dir / scene, pc)
                if visualize:
                    visualize_pointcloud(pc, output_dir / f"{scene}.obj")


def sample_scene_point_clouds(config, full_scene_dir, num_points, output_dir, visualize):
    from util.visualization import visualize_sdf_as_mesh
    sigma = 0.25
    split_shapes = read_list(f"{config['dataset_train']['data_dir']}/splits/{config['dataset_train']['dataset_name']}/{config['dataset_train']['splits_dir']}/val.txt")
    # split_shapes.extend(read_list(f"{config['dataset_train']['data_dir']}/splits/{config['dataset_train']['dataset_name']}/{config['dataset_train']['splits_dir']}/val.txt"))
    split_shapes = list(set(split_shapes))
    all_scenes = list(set(["__".join(s.split('__')[:3]) for s in split_shapes]))

    for scene in tqdm(all_scenes):
        if Path(full_scene_dir, scene + ".npy").exists():
            if (Path(output_dir) / (scene + ".npz")).exists():
                continue
            scene_df = np.load(Path(full_scene_dir, scene + ".npy"))
            num_chunks = len([x for x in split_shapes if x.startswith(scene)])
            num_points_to_sample = num_chunks * num_points
            tmp_mesh_path = Path("/tmp", scene + '.obj')
            visualize_sdf_as_mesh(scene_df, tmp_mesh_path, 0.75 * config['dataset_train']['voxel_size_target'])
            mesh = trimesh.load(tmp_mesh_path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                continue
            points_surface, _ = sample.sample_surface(mesh, num_points_to_sample // 2)
            points_jittered, _ = sample.sample_surface(mesh, num_points_to_sample * 4)
            points_jittered = points_jittered + sigma * np.random.randn(points_jittered.shape[0], 3)
            points_grid = np.clip(points_jittered, 0, scene_df.shape[0] - 1).astype(np.uint32)
            points_jittered_occupied = scene_df[points_grid[:, 0], points_grid[:, 1], points_grid[:, 2]] <= 0.75 * config['dataset_train']['voxel_size_target']
            points_jittered = points_jittered[points_jittered_occupied]
            if points_jittered.shape[0] > num_points_to_sample - num_points_to_sample // 2:
                points_jittered = points_jittered[random.sample(range(points_jittered.shape[0]), num_points_to_sample - num_points_to_sample // 2), :]
            all_points = np.concatenate([points_surface, points_jittered], axis=0)
            # print(num_chunks, all_points.shape[0])
            os.remove(tmp_mesh_path)
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            np.savez_compressed(Path(output_dir) / scene, all_points)
            if visualize:
                visualize_pointcloud(all_points, Path(output_dir) / f"{scene}.obj")
        else:
            print(full_scene_dir, scene + ".npy")


def visualize_retrievals(path_to_retrievals, sample_name, voxel_size):
    list_of_npzs = Path(path_to_retrievals).iterdir()
    positions = []
    chunks = []
    for x in list_of_npzs:
        if x.name.startswith(sample_name):
            positions.append([int(y) for y in x.name.split(".")[0].split("__")[-1].split("_")])
            chunks.append(np.load(x)["arr_0"])
    combined = np.ones([8, np.array(positions)[:, 0].max() + 64, np.array(positions)[:, 1].max() + 64, np.array(positions)[:, 2].max() + 64]) * voxel_size * 3
    for k in range(8):
        for i, c in enumerate(chunks):
            combined[k, positions[i][0]:positions[i][0] + 64, positions[i][1]:positions[i][1] + 64, positions[i][2]:positions[i][2] + 64] = c[k]
        visualize_sdf_as_mesh(combined[k], f"{sample_name}_nn{k + 1}.obj", voxel_size * 0.75)
