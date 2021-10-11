from collections import defaultdict

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from pathlib import Path
from util.misc import read_list
import numpy as np

from dataset.scene import SceneHandler


class PatchedSceneDataset(Dataset):

    def __init__(self, split, dataset_config, scene_handler):
        self.scene_handler = scene_handler
        self.dataset_name = dataset_config['dataset_name']
        self.input_mean, self.input_std = dataset_config['input_mean'], dataset_config['input_std']
        self.target_mean, self.target_std = dataset_config['target_mean'], dataset_config['target_std']
        self.use_retrievals = scene_handler.use_retrievals
        self.scenes = read_list(f"{dataset_config['data_dir']}/splits/{dataset_config['dataset_name']}/{dataset_config['splits_dir']}/{split}.txt")
        self.scenes = [x for x in self.scenes if Path(dataset_config['data_dir'], dataset_config['target_dir'], dataset_config['dataset_name'], (x + dataset_config['target_ext'])).exists()]
        self.scenes = [x for x in self.scenes if Path(dataset_config['data_dir'], dataset_config['input_dir'], dataset_config['dataset_name'], (x + dataset_config['input_ext'])).exists()]
        self.data = []
        for s in tqdm(self.scenes, desc='psh_read'):
            input_extent, target_extent = self.scene_handler.get_scene_patches(s)
            for ii in range(len(input_extent)):
                if self.scene_handler.get_patch_occupancy(s, target_extent[ii]) > dataset_config['occupancy_threshold']:
                    self.data.append([s, input_extent[ii], target_extent[ii]])
        self.patch_from_scene_lookup = defaultdict(list)
        for d in self.data:
            self.patch_from_scene_lookup[d[0]].append(SceneHandler.get_name_from_extent(d[0], d[2]))
        if split == 'train':
            self.data = self.data * dataset_config['train_multiplier']

    def use_subset(self, subset):
        new_data = []
        subset_extent = [self.scene_handler.get_extent_from_name(x) for x in subset]
        for d in subset_extent:
            new_data.append([d[0], [int(e // self.scene_handler.scale_factor) for e in d[1]], d[1]])
        self.data = new_data

    @property
    def target_trunc(self):
        return self.scene_handler.target_trunc

    @property
    def target_voxel_size(self):
        return self.scene_handler.target_voxel_size

    @property
    def input_trunc(self):
        return self.scene_handler.input_trunc

    @property
    def input_voxel_size(self):
        return self.scene_handler.input_voxel_size

    @property
    def target_patch_size(self):
        return self.scene_handler.patch_size_target

    @property
    def target_patch_context(self):
        return self.scene_handler.patch_context_target

    @property
    def input_chunk_size(self):
        return self.scene_handler.input_chunk_size

    @property
    def target_chunk_size(self):
        return self.scene_handler.target_chunk_size

    def get_scene_size(self, scene):
        return self.scene_handler.scene_size[scene]

    def get_scene_indices(self, scenes):
        return np.array([self.scenes.index(s) for s in scenes])

    def get_scene_names_from_patches(self, patch_names):
        return [self.scene_handler.get_extent_from_name(x)[0] for x in patch_names]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_scene_unpadded(scene, scene_handler_func, patch_context):
        scene_padded = scene_handler_func(scene)
        unpad_0, unpad_1 = patch_context, scene_padded.shape[0] - patch_context
        unpad_2, unpad_3 = patch_context, scene_padded.shape[1] - patch_context
        unpad_4, unpad_5 = patch_context, scene_padded.shape[2] - patch_context
        return scene_padded[unpad_0: unpad_1, unpad_2: unpad_3, unpad_4: unpad_5]

    def get_scene_input(self, scene):
        return PatchedSceneDataset.get_scene_unpadded(scene, self.scene_handler.get_scene_input, self.scene_handler.patch_context_input)

    def get_scene_target(self, scene):
        return PatchedSceneDataset.get_scene_unpadded(scene, self.scene_handler.get_scene_target, self.scene_handler.patch_context_target)

    def unpad(self, *extents):
        if len(extents) == 2:
            return [extents[0], extents[1] - 2 * self.scene_handler.patch_context_target]
        else:
            return self.unpad(extents[0], extents[1]) + self.unpad(extents[2], extents[3]) + self.unpad(extents[4], extents[5])

    def pad(self, *extents):
        if len(extents) == 2:
            return [extents[0], extents[1] + 2 * self.scene_handler.patch_context_target]
        else:
            return self.pad(extents[0], extents[1]) + self.pad(extents[2], extents[3]) + self.pad(extents[4], extents[5])

    @property
    def no_overlap(self):
        return self.scene_handler.patch_stride_target == self.scene_handler.patch_size_target

    def __getitem__(self, index):
        item_data = self.data[index]
        scene_shape_input = self.scene_handler.get_scene_input(item_data[0])
        scene_shape_target = self.scene_handler.get_scene_target(item_data[0])
        patch_input = scene_shape_input[item_data[1][0]: item_data[1][1], item_data[1][2]: item_data[1][3], item_data[1][4]: item_data[1][5]]
        patch_target = scene_shape_target[item_data[2][0]: item_data[2][1], item_data[2][2]: item_data[2][3], item_data[2][4]: item_data[2][5]]
        return_dict = {
            'name': SceneHandler.get_name_from_extent(item_data[0], item_data[2]),
            'scene': item_data[0],
            'extent': item_data[2],
            'input': (patch_input[np.newaxis, ...] - self.input_mean) / self.input_std,
            'target': (patch_target[np.newaxis, ...] - self.target_mean) / self.target_std,
        }
        if self.use_retrievals:
            scene_shape_retrieval = self.scene_handler.get_scene_retrieval(item_data[0])
            patch_retrieval = scene_shape_retrieval[:, item_data[2][0]: item_data[2][1], item_data[2][2]: item_data[2][3], item_data[2][4]: item_data[2][5]]
            return_dict['retrieval'] = (patch_retrieval - self.target_mean) / self.target_std
        else:
            patch_retrieval = np.ones((4, item_data[2][1] - item_data[2][0], item_data[2][3] - item_data[2][2], item_data[2][5] - item_data[2][4]), dtype=np.float32) * self.target_trunc
            return_dict['retrieval'] = patch_retrieval
        return return_dict

    def compute_normals(self, target):
        padded_array = torch.nn.functional.pad(target, [1, 1, 1, 1, 1, 1], mode='constant', value=self.scene_handler.target_trunc)
        dx = torch.nn.functional.conv3d(padded_array, self.sobel_3d_x.to(padded_array.device))
        dy = torch.nn.functional.conv3d(padded_array, self.sobel_3d_y.to(padded_array.device))
        dz = torch.nn.functional.conv3d(padded_array, self.sobel_3d_z.to(padded_array.device))
        normals = torch.cat((dx, dy, dz), dim=1)
        normalizer = torch.sqrt(torch.square(normals).sum(dim=1, keepdim=True) + 1e-5)
        return torch.div(normals, normalizer)

    def compute_laplacian(self, target):
        padded_array = torch.nn.functional.pad(target, [1, 1, 1, 1, 1, 1], mode='constant', value=self.scene_handler.target_trunc)
        laplacian = torch.nn.functional.conv3d(padded_array, self.laplacian_3d.to(padded_array.device))
        return laplacian

    def get_superscene_name_and_position_from_chunk(self, chunk_name):
        if self.dataset_name.startswith('Matterport3D') or self.dataset_name.startswith('3DFront'):
            name = "__".join(chunk_name.split('__')[:2])
            position = [int(x) for x in chunk_name.split('__')[-1].split('_')]
            return name, np.array(position)
        return chunk_name, np.array([0, 0, 0])

    def combine_chunks(self, scale_factor, chunk_size, trunc_val, scene_accessor, container_obj):
        result = {}
        superscene_chunks = defaultdict(list)
        for s in self.scenes:
            name, position = self.get_superscene_name_and_position_from_chunk(s)
            superscene_chunks[name].append((s, (position / scale_factor).astype(np.int32)))
        for ss in superscene_chunks:
            chunkpositions = superscene_chunks[ss]
            positions = np.vstack([cp[1] for cp in chunkpositions])
            combined = np.ones([positions[:, 0].max() + chunk_size, positions[:, 1].max() + chunk_size, positions[:, 2].max() + chunk_size]) * trunc_val
            for cp in chunkpositions:
                scene_unpadded = scene_accessor(container_obj, cp[0])
                combined[cp[1][0]:cp[1][0] + scene_unpadded.shape[0], cp[1][1]:cp[1][1] + scene_unpadded.shape[0], cp[1][2]:cp[1][2] + scene_unpadded.shape[0]] = scene_unpadded
            result[ss] = combined
        return result

    def combine_inputs(self):
        return self.combine_chunks(self.target_chunk_size / self.input_chunk_size, self.input_chunk_size, self.input_trunc, PatchedSceneDataset.get_scene_input, self)

    def combine_targets(self):
        return self.combine_chunks(1, self.target_chunk_size, self.target_trunc, PatchedSceneDataset.get_scene_target, self)

    def combine_retrievals(self, retrievals, k):
        def accessor(passed_obj, name):
            _retrievals, _scenes, _k = passed_obj
            return _retrievals[_scenes.index(name), k, :, :, :]
        return self.combine_chunks(1, self.target_chunk_size, self.target_trunc, accessor, [retrievals, self.scenes, k])

    def denormalize_target(self, patch):
        return patch * self.target_std + self.target_mean

    def denormalize_input(self, patch):
        return patch * self.input_std + self.input_mean

    sobel_3d_x = torch.from_numpy(np.array([[[+1, +2, +1], [+2, +4, +2], [+1, +2, +1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_3d_y = torch.from_numpy(np.array([[[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], [[+2, +4, +2], [0, 0, 0], [-2, -4, -2]], [[+1, +2, +1], [0, 0, 0], [-1, -2, -1]]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    sobel_3d_z = torch.from_numpy(np.array([[[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], [[-2, 0, +2], [-4, 0, +4], [-2, 0, +2]], [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]], dtype=np.float32)).unsqueeze(0).unsqueeze(0)

    laplacian_3d = torch.from_numpy(np.array([[[2, 3, 2], [3, 6, 3], [2, 3, 2]], [[3, 6, 3], [6, -88, 6], [3, 6, 3]], [[2, 3, 2], [3, 6, 2], [2, 3, 2]]], dtype=np.float32)).unsqueeze(0).unsqueeze(0) / 26


class CombinedDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = datasets
        self.scenes = []
        for ds in self.datasets:
            self.scenes.extend(ds.scenes)

    def __len__(self):
        total_len = 0
        for ds in self.datasets:
            total_len += len(ds)
        return total_len

    def __getitem__(self, index):
        offset = 0
        item = None
        for ds in self.datasets:
            if index < len(ds) + offset:
                item = ds[index - offset]
                break
            offset += len(ds)
        item['input'] = []
        return item

    def get_scene_indices(self, scenes):
        return np.array([self.scenes.index(s) for s in scenes])

    def unpad(self, *extents):
        return self.datasets[0].unpad(*extents)

    @property
    def target_patch_size(self):
        return self.datasets[0].target_patch_size

    @property
    def target_patch_context(self):
        return self.datasets[0].target_patch_context

    def get_scene_target(self, scene):
        for ds in self.datasets:
            if scene in ds.scenes:
                return ds.get_scene_target(scene) * self.datasets[0].target_voxel_size / ds.target_voxel_size
        raise NotImplementedError
