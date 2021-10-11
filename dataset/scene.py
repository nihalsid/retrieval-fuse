from pathlib import Path
import json
import numpy as np
import torch
import random
from util.misc import read_list, point_cloud_to_grid
from tqdm import tqdm

from util.misc import get_retrievals_dir
from util.visualization import visualize_sdf_as_mesh, visualize_sdf_as_voxels, visualize_grid_as_voxels, visualize_float_grid, visualize_normals


class SceneHandler:

    def __init__(self, split, config):
        self.task = config['task']
        self.scene_size = {}
        self.scene_occupancy = {}
        self.preloaded_scenes_input = {}
        self.preloaded_scenes_target = {}
        self.preloaded_retrievals = {}
        self.random_indices_list = None
        self.split_shapes = []
        self.retrievals_dir = None
        self.fast_visualization = config['fast_visualization']
        dataset_config = config[f'dataset_{split}']
        self.input_chunk_size = dataset_config['input_chunk_size']
        self.target_chunk_size = dataset_config['target_chunk_size']
        self.number_point_samples = dataset_config['num_points']
        self.input_voxel_size = np.float16(dataset_config['voxel_size_input']).astype(np.float32)
        self.target_voxel_size = np.float16(dataset_config['voxel_size_target']).astype(np.float32)
        self.input_trunc = np.float16(dataset_config['voxel_size_input'] * 3).astype(np.float32)
        self.target_trunc = np.float16(dataset_config['voxel_size_target'] * 3).astype(np.float32)
        self.patch_size_target = dataset_config['patch_size_target']
        self.patch_context_target = dataset_config['patch_context_target']
        self.patch_stride_target = dataset_config['patch_stride']
        self.patch_size_input = dataset_config['patch_size_input']
        self.patch_context_input = dataset_config['patch_context_input']
        self.patch_stride_input = int(dataset_config['patch_stride'] * dataset_config['patch_size_input'] / dataset_config['patch_size_target'])
        self.scale_factor = dataset_config['patch_size_target'] / dataset_config['patch_size_input']
        self.input_ext = dataset_config['input_ext']
        self.target_ext = dataset_config['target_ext']
        self.input_path = Path(dataset_config['scene_dir'], dataset_config['input_dir'], dataset_config['dataset_name'])
        self.target_path = Path(dataset_config['scene_dir'], dataset_config['target_dir'], dataset_config['dataset_name'])
        self.input_loader = self.pc_loader if self.task == 'surface_reconstruction' else self.df_loader
        self.get_scene_input = self.get_pc_scene_input if self.task == 'surface_reconstruction' else self.get_df_scene_input
        self.visualize_input_chunk = visualize_grid_as_voxels if self.task == 'surface_reconstruction' else self.visualize_input_chunk_df
        self.split_shapes.extend(read_list(f"{dataset_config['data_dir']}/splits/{dataset_config['dataset_name']}/{dataset_config['splits_dir']}/{split}.txt"))
        self.scenes = [x for x in self.split_shapes]
        self.use_retrievals = not config['no_retrievals']
        if self.use_retrievals:
            print('Using retrievals.')
            self.retrievals_dir = get_retrievals_dir(config)
        self.load_to_memory(dataset_config["preload_scenes"], dataset_config["preload_retrievals"])
        self.initialize_random_indices_list(Path(dataset_config['data_dir'], "random_indices", f"{self.number_point_samples}.npz"))
        self.initialize_scene_sizes(Path(dataset_config['data_dir'], 'size', dataset_config['dataset_name'] + '.json'))
        if not dataset_config['skip_occupancy']:
            self.initialize_scene_occupancy(Path(dataset_config['data_dir'], 'occupancy', f"{dataset_config['dataset_name']}_{self.target_chunk_size:03d}_{self.patch_size_target:02d}_{self.patch_context_target:02d}.json"))

    def df_loader(self, scene):
        return np.pad(np.load(self.input_path / (scene + self.input_ext))["arr"].astype(np.float16), self.patch_context_input, mode='constant', constant_values=self.input_trunc)

    def pc_loader(self, scene):
        return np.load(self.input_path / (scene + self.input_ext))["arr_0"]

    def load_to_memory(self, preload_scenes, preload_retrievals):
        if preload_scenes:
            for s in tqdm(self.scenes, 'sh_preload'):
                # noinspection PyArgumentList
                self.preloaded_scenes_input[s] = self.input_loader(s)
                self.preloaded_scenes_target[s] = np.pad(np.load(self.target_path / (s + self.target_ext))["arr"].astype(np.float16), self.patch_context_target, mode='constant', constant_values=self.target_trunc)
        if self.use_retrievals and preload_retrievals:
            for s in tqdm(self.scenes, 'rtr_preload'):
                self.preloaded_retrievals[s] = np.pad(np.load(self.retrievals_dir / "compose" / (s + '.npz'))["arr_0"].astype(np.float16), self.patch_context_target, mode='constant', constant_values=self.target_trunc)

    def get_df_scene_input(self, scene):
        if scene not in self.preloaded_scenes_input:
            return self.df_loader(scene).astype(np.float32)
        return self.preloaded_scenes_input[scene].astype(np.float32)

    def get_pc_scene_input(self, scene):
        if scene not in self.preloaded_scenes_input:
            pc = self.pc_loader(scene)
        else:
            pc = self.preloaded_scenes_input[scene]
        if pc.shape[0] < 20000:
            pc = np.vstack([pc, pc])
        pt_indices = self.random_indices_list[random.randint(0, self.random_indices_list.shape[0] - 1)]
        pc = pc[pt_indices, :]
        return point_cloud_to_grid(pc, self.input_chunk_size, 1 / self.scale_factor, self.patch_context_input)

    def get_scene_target(self, scene):
        if scene not in self.preloaded_scenes_target:
            return np.pad(np.load(self.target_path / (scene + self.target_ext))["arr"].astype(np.float32), self.patch_context_target, mode='constant', constant_values=self.target_trunc)
        return self.preloaded_scenes_target[scene].astype(np.float32)

    def get_scene_retrieval(self, scene):
        if scene not in self.preloaded_retrievals:
            return np.pad(np.load(self.retrievals_dir / "compose" / (scene + '.npz'))["arr_0"].astype(np.float32), self.patch_context_target, mode='constant', constant_values=self.target_trunc)
        return self.preloaded_retrievals[scene].astype(np.float32)

    def initialize_random_indices_list(self, filepath):
        if filepath.exists():
            self.random_indices_list = np.load(filepath)["arr"]
        else:
            rand_list_size = 20000 * 10
            temp_list = [None] * rand_list_size
            for i in tqdm(list(range(rand_list_size))):
                temp_list[i] = random.sample(range(20000), self.number_point_samples)
            self.random_indices_list = np.array(temp_list)
            filepath.parents[0].mkdir(exist_ok=True)
            np.savez_compressed(filepath, arr=self.random_indices_list)

    def initialize_scene_sizes(self, filepath):
        needs_recreation = not filepath.exists()
        if filepath.exists():
            self.scene_size = json.loads(filepath.read_text())
            for scene in self.scenes:
                if scene not in self.scene_size:
                    needs_recreation = True
                    break
        if needs_recreation:
            for scene in tqdm(self.scenes, desc='save_sizes'):
                self.scene_size[scene] = [s - 2 * self.patch_context_target for s in self.get_scene_target(scene).shape]
            filepath.parents[0].mkdir(exist_ok=True)
            filepath.write_text(json.dumps(self.scene_size))

    def initialize_scene_occupancy(self, filepath):
        needs_recreation = not filepath.exists()
        if filepath.exists():
            self.scene_occupancy = json.loads(filepath.read_text())
            for scene in self.scenes:
                _, target_extents = self.get_scene_patches(scene)
                for t_ext_idx in range(target_extents.shape[0]):
                    name = SceneHandler.get_name_from_extent(scene, target_extents[t_ext_idx, :])
                    if name not in self.scene_occupancy:
                        needs_recreation = True
                        break
        if needs_recreation:
            for scene in tqdm(self.scenes, desc='save_occ'):
                _, target_extents = self.get_scene_patches(scene)
                for t_ext_idx in range(target_extents.shape[0]):
                    name = SceneHandler.get_name_from_extent(scene, target_extents[t_ext_idx, :])
                    self.scene_occupancy[name] = self.calculate_occupancy_for_name(name)
            filepath.parents[0].mkdir(exist_ok=True)
            filepath.write_text(json.dumps(self.scene_occupancy))

    def calculate_occupancy_for_name(self, patch_identifier):
        scene, extent = SceneHandler.get_extent_from_name(patch_identifier)
        return int((self.get_scene_target(scene)[extent[0]:extent[1], extent[2]:extent[3], extent[4]:extent[5]] <= 0.75 * 2 * self.target_voxel_size).sum())

    @staticmethod
    def get_extents_for_size(size, patch_size, patch_context, patch_stride):
        end_point = lambda x: x - patch_size
        lx = np.linspace(0, end_point(size[0]), end_point(size[0]) // patch_stride + 1).astype(np.int32)
        ly = np.linspace(0, end_point(size[1]), end_point(size[1]) // patch_stride + 1).astype(np.int32)
        lz = np.linspace(0, end_point(size[2]), end_point(size[2]) // patch_stride + 1).astype(np.int32)
        x_start, y_start, z_start = np.meshgrid(lx, ly, lz, indexing='ij')
        x_end, y_end, z_end = x_start + patch_size + 2 * patch_context, y_start + patch_size + 2 * patch_context, z_start + patch_size + 2 * patch_context
        return np.hstack((x_start.flatten()[:, np.newaxis], x_end.flatten()[:, np.newaxis], y_start.flatten()[:, np.newaxis], y_end.flatten()[:, np.newaxis], z_start.flatten()[:, np.newaxis], z_end.flatten()[:, np.newaxis]))

    def get_scene_patches(self, scene):
        size_target = self.scene_size[scene]
        size_input = [int(s / self.scale_factor) for s in self.scene_size[scene]]
        extents_target = self.get_extents_for_size(size_target, self.patch_size_target, self.patch_context_target, self.patch_stride_target)
        extents_input = self.get_extents_for_size(size_input, self.patch_size_input, self.patch_context_input, self.patch_stride_input)
        return extents_input, extents_target

    @staticmethod
    def get_name_from_extent(scene, extent_target):
        return f"{scene}--{extent_target[0]:04d}_{extent_target[1]:04d}_{extent_target[2]:04d}_{extent_target[3]:04d}_{extent_target[4]:04d}_{extent_target[5]:04d}"

    @staticmethod
    def get_extent_from_name(identifier):
        scene, rest = identifier.split('--')
        extent = [int(r) for r in rest.split('_')]
        return scene, extent

    def create_scene_volume_from_extents(self, scene, occupancy_threshold=0):
        size = list(map(lambda x: x + 2 * self.patch_context_target, self.scene_size[scene]))
        df_volume_input = np.ones(list(map(lambda x: int(x / self.scale_factor), size)), dtype=np.float32) * self.input_trunc
        df_volume_target = np.ones(size, dtype=np.float32) * self.target_trunc
        patches_input, patches_target = self.get_scene_patches(scene)
        # noinspection PyArgumentList
        input_scene = self.get_scene_input(scene)
        target_scene = self.get_scene_target(scene)
        for pidx in range(patches_input.shape[0]):
            name = SceneHandler.get_name_from_extent(scene, patches_target[pidx, :])
            if self.scene_occupancy[name] >= occupancy_threshold:
                src_volume_input = input_scene[patches_input[pidx, 0]: patches_input[pidx, 1], patches_input[pidx, 2]: patches_input[pidx, 3], patches_input[pidx, 4]: patches_input[pidx, 5]]
                src_volume_target = target_scene[patches_target[pidx, 0]: patches_target[pidx, 1], patches_target[pidx, 2]: patches_target[pidx, 3], patches_target[pidx, 4]: patches_target[pidx, 5]]
                df_volume_input[patches_input[pidx, 0]: patches_input[pidx, 1], patches_input[pidx, 2]: patches_input[pidx, 3], patches_input[pidx, 4]: patches_input[pidx, 5]] = src_volume_input
                df_volume_target[patches_target[pidx, 0]: patches_target[pidx, 1], patches_target[pidx, 2]: patches_target[pidx, 3], patches_target[pidx, 4]: patches_target[pidx, 5]] = src_volume_target
        assert np.abs(df_volume_input - input_scene).mean() < 1e-5
        assert np.abs(df_volume_target - target_scene).mean() < 1e-5
        return df_volume_input, df_volume_target

    def get_all_patches_of_size(self, size):
        pruned_patches = {}
        for patch in self.scene_occupancy.keys():
            _, extent = SceneHandler.get_extent_from_name(patch)
            if (extent[1] - extent[0]) == size and (extent[3] - extent[2]) == size and (extent[5] - extent[4]) == size:
                pruned_patches[patch] = self.scene_occupancy[patch]
        return pruned_patches

    def get_patch_occupancy(self, scene, target_extent):
        name = SceneHandler.get_name_from_extent(scene, target_extent)
        if name in self.scene_occupancy:
            return self.scene_occupancy[name]
        else:
            return 1

    def visualize_target_chunk(self, chunk_df, output_path):
        scale_factor = 1
        if not self.fast_visualization:
            chunk_df = torch.nn.functional.interpolate(torch.from_numpy(chunk_df).cuda().unsqueeze(0).unsqueeze(0), scale_factor=2, mode='trilinear', align_corners=True).squeeze(0).squeeze(0).cpu().numpy()
            scale_factor = 2
        visualize_sdf_as_mesh(chunk_df, output_path, self.target_voxel_size * 0.75, scale_factor=scale_factor)

    def visualize_input_chunk_df(self, chunk_df, output_path):
        visualize_sdf_as_voxels(chunk_df, output_path, self.input_voxel_size * 0.675)

    @staticmethod
    def visualize_weight(chunk_weight, output_path):
        visualize_float_grid(chunk_weight, 1, 1, 4, output_path)

    @staticmethod
    def visualize_normal(chunk_normal, output_path):
        visualize_normals(chunk_normal, output_path)
