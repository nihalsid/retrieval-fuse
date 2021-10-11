import traceback

import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from pyflann import *
import torch.utils.data

from config import config_handler
from dataset.patched_scene_dataset import PatchedSceneDataset, CombinedDataset
from dataset.scene import SceneHandler
from model import get_retrieval_networks
from util.metrics import IoU, Precision, Recall, Chamfer3D
from util.misc import tensor3d_to_tensor5d, array3d_to_tensor5d, load_net_and_set_to_eval, get_retrievals_dir
from util.timer import Timer
import multiprocessing


def get_zero_patch_entry(feature_extractor, patch_size, patch_context, latent_dim):
    with torch.no_grad():
        target_patch = torch.from_numpy(np.ones([patch_size + 2 * patch_context] * 3, dtype=np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        prediction = feature_extractor(target_patch)
        prediction = torch.nn.functional.normalize(prediction.permute((0, 2, 3, 4, 1)).reshape((-1, latent_dim)), dim=1).cpu().numpy()
    return np.hstack([np.array([-1], dtype=np.float32)[:, np.newaxis]] + [np.array([0], dtype=np.float32)[:, np.newaxis], np.array([patch_size], dtype=np.float32)[:, np.newaxis]] * 3 + [prediction])


def create_dictionary(feature_extractor, dictionary_config, latent_dim, dataset, tree_path):
    tree_path.mkdir(exist_ok=True, parents=True)
    number_of_patches = len(dataset)
    database = np.zeros((number_of_patches + 1, 1 + 3 * 2 + latent_dim), dtype=np.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dictionary_config["batch_size"], shuffle=False, num_workers=dictionary_config["num_workers"], drop_last=False)
    with torch.no_grad():
        for i, item in enumerate(tqdm(dataloader, desc='dict_feats')):
            target_patches = item['target'].cuda()
            prediction = feature_extractor(target_patches)
            prediction = torch.nn.functional.normalize(prediction.permute((0, 2, 3, 4, 1)).reshape((-1, latent_dim)), dim=1).cpu().numpy()
            db_start_idx, db_end_idx = i * dictionary_config["batch_size"], i * dictionary_config["batch_size"] + target_patches.shape[0]
            scene_index = dataset.get_scene_indices(item['scene'])[:, np.newaxis]
            x_start, x_end = dataset.unpad(item['extent'][:, 0].cpu().numpy(), item['extent'][:, 1].cpu().numpy())
            y_start, y_end = dataset.unpad(item['extent'][:, 2].cpu().numpy(), item['extent'][:, 3].cpu().numpy())
            z_start, z_end = dataset.unpad(item['extent'][:, 4].cpu().numpy(), item['extent'][:, 5].cpu().numpy())
            database[db_start_idx: db_end_idx, :] = np.hstack([scene_index, x_start[:, np.newaxis], x_end[:, np.newaxis], y_start[:, np.newaxis], y_end[:, np.newaxis], z_start[:, np.newaxis], z_end[:, np.newaxis], prediction])
    database[number_of_patches, :] = get_zero_patch_entry(feature_extractor, dataset.target_patch_size, dataset.target_patch_context, latent_dim)
    with Timer("Dictionary Indexing"):
        np.save(tree_path / "database", database)
        Path(tree_path / "index.json").write_text(json.dumps(dataset.scenes))
        flann_obj = FLANN()
        params = flann_obj.build_index(database[:, 7:], algorithm="kdtree", trees=64, log_level="info")
        # nihalsid: some past algo's we tried. autotune is too slow to be usable, linear gets unusable when too many patches - not sure if its really bruteforce
        # params = flann_obj.build_index(database[:, 7:], algorithm="autotuned", target_precision=1, log_level="info")
        # params = flann_obj.build_index(database[:, 7:], algorithm="linear", target_precision=1.0, log_level="info")
        Path(tree_path / "params.json").write_text(json.dumps(params))
        flann_obj.save_index((str(tree_path / "index_010_64_tree.idx")).encode('utf-8'))


def extract_features(feature_extractor, query_config, latent_dim, dataset, key):
    features = np.zeros((len(dataset), latent_dim), dtype=np.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=query_config["batch_size"], shuffle=False, num_workers=query_config["num_workers"], drop_last=False)
    patch_names = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(dataloader, desc='query_features')):
            patch_names.extend(item['name'])
            item[key] = item[key].cuda()
            normalized_features = torch.nn.functional.normalize(feature_extractor(item[key]).permute((0, 2, 3, 4, 1)).reshape((-1, latent_dim)), dim=1).cpu().numpy()
            features[i * query_config["batch_size"]: i * query_config["batch_size"] + item[key].shape[0], :] = normalized_features
    return patch_names, features


def extract_input_features(feature_extractor, query_config, latent_dim, dataset):
    return extract_features(feature_extractor, query_config, latent_dim, dataset, 'input')


def extract_target_features(feature_extractor, query_config, latent_dim, dataset):
    return extract_features(feature_extractor, query_config, latent_dim, dataset, 'target')


def flann_knn_worker(results_dict, K, tree_path, worker_scene_names, worker_patch_names, worker_features, ignore_patches_from_source):
    try:
        flann_obj = FLANN()
        database = np.load(tree_path / "database.npy")
        flann_obj.load_index(str(tree_path / "index_010_64_tree.idx").encode('utf-8'), database[:, 7:])
        dataset_index = json.loads((tree_path / "index.json").read_text())
        params = json.loads((tree_path / "params.json").read_text())

        query_batch_size = 1024
        for batch_idx in tqdm(range(worker_features.shape[0] // query_batch_size + 1), 'flann_knn_worker'):
            feature_subset = worker_features[batch_idx * query_batch_size: (batch_idx + 1) * query_batch_size, :]
            worker_scene_name_subset = worker_scene_names[batch_idx * query_batch_size: (batch_idx + 1) * query_batch_size]
            worker_patch_name_subset = worker_patch_names[batch_idx * query_batch_size: (batch_idx + 1) * query_batch_size]
            results, dists = flann_obj.nn_index(feature_subset, 2 * K, checks=params['checks'])
            all_extents = np.stack([np.hstack((database[np.array(results[:, i]), 0:7], np.array(dists[:, i])[:, None])) for i in range(2 * K)]).transpose((1, 0, 2))
            for i in range(all_extents.shape[0]):
                if ignore_patches_from_source and worker_scene_name_subset[i] in dataset_index:
                    M = all_extents[i, :, 0] == dataset_index.index(worker_scene_name_subset[i])
                    all_extents[i, :, :] = np.concatenate((all_extents[i, ~M, :], all_extents[i, M, :]))
            all_extents = all_extents.transpose((1, 0, 2))
            for i, cn in enumerate(worker_patch_name_subset):
                results_dict[cn] = all_extents[:K, i, :]
                # if all_extents[:K, i, 0].max() >= len(dataset_index):
                #     print("ERROR: ", cn, all_extents[:K, i, 0])
    except Exception as err:
        print("FLANN Worker failed with exception", err)
        traceback.print_exc()


def query_dictionary_using_features_parallel(query_config, tree_path, input_scene_names, input_patch_names, input_features, ignore_patches_from_source):
    manager = multiprocessing.Manager()
    # noinspection PyTypeChecker
    shared_dict = manager.dict([(input_patch_names[i], None) for i in range(len(input_patch_names))])
    items_per_worker = input_features.shape[0] // query_config["flann_num_workers"] + 1
    process = []
    for pid in range(query_config["flann_num_workers"]):
        worker_scene_names = input_scene_names[pid * items_per_worker: (pid + 1) * items_per_worker]
        worker_patch_names = input_patch_names[pid * items_per_worker: (pid + 1) * items_per_worker]
        worker_features = input_features[pid * items_per_worker: (pid + 1) * items_per_worker, :]
        process.append(multiprocessing.Process(target=flann_knn_worker, args=(shared_dict, query_config["K"], tree_path, worker_scene_names, worker_patch_names, worker_features, ignore_patches_from_source)))
    for p in process:
        p.start()
    for p in process:
        p.join()
    total_items = len(input_patch_names)
    total_mapped = 0
    retrieval_mapping = {}
    for item in shared_dict:
        retrieval_mapping[item] = shared_dict[item]
        if shared_dict[item] is not None:
            total_mapped += 1
    print(f"{total_mapped}/{total_items} mapped")
    return retrieval_mapping


def query_dictionary_using_features(query_config, patch_names, input_features, dataset, tree_path, ignore_patches_from_source):
    scene_names = dataset.get_scene_names_from_patches(patch_names)
    with Timer("FLANN"):
        if query_config["flann_num_workers"] == 0:
            retrieval_mapping = dict.fromkeys(patch_names)
            flann_knn_worker(retrieval_mapping, query_config["K"], tree_path, scene_names, patch_names, input_features, ignore_patches_from_source)
        else:
            retrieval_mapping = query_dictionary_using_features_parallel(query_config, tree_path, scene_names, patch_names, input_features, ignore_patches_from_source)
    return retrieval_mapping


def create_retrieval_from_mapping(scene_name, retrieval_mappings, K, dataset_train, dataset, tree_path):
    dataset_index = json.loads((tree_path / "index.json").read_text())
    scene_size = dataset.get_scene_size(scene_name)
    scene_retrieval = torch.from_numpy(np.ones((K, scene_size[0], scene_size[1], scene_size[2]), dtype=np.float32)) * dataset.target_trunc
    distances = torch.ones_like(scene_retrieval) * 100
    all_patches_for_scene = dataset.patch_from_scene_lookup[scene_name]
    for k in range(K):
        for p in all_patches_for_scene:
            X0, X1, Y0, Y1, Z0, Z1 = retrieval_mappings[p][k, 1:7].astype(np.int32).tolist()
            current_distance = retrieval_mappings[p][k, 7]
            xx0, xx1, yy0, yy1, zz0, zz1 = dataset_train.unpad(*SceneHandler.get_extent_from_name(p)[1])
            if dataset.no_overlap or distances[k, xx0: xx1, yy0: yy1, zz0: zz1].mean() > current_distance:
                index_ptr = int(retrieval_mappings[p][k, 0])
                if index_ptr >= 0:
                    shape = torch.from_numpy(dataset_train.get_scene_target(dataset_index[index_ptr]))
                else:
                    shape = torch.from_numpy(np.ones((scene_size[0], scene_size[1], scene_size[2])) * dataset.target_trunc)
                scene_retrieval[k, xx0: xx1, yy0: yy1, zz0: zz1] = shape[X0:X1, Y0:Y1, Z0:Z1] * (dataset.target_trunc / dataset_train.target_trunc)
                distances[k, xx0: xx1, yy0: yy1, zz0: zz1] = float(current_distance)
    return scene_retrieval


def get_metrics_for_retrieval(retrievals, dataset):
    metrics = [IoU(compute_on_step=False).cuda(), Chamfer3D(compute_on_step=False).cuda(), Precision(compute_on_step=False).cuda(), Recall(compute_on_step=False).cuda()]
    for idx, scene in enumerate(tqdm(dataset.scenes, 'metric')):
        retrieved_scene = retrievals[idx]
        nn1 = tensor3d_to_tensor5d((retrieved_scene[0] <= 0.75 * dataset.target_voxel_size)).cuda()
        target = array3d_to_tensor5d((dataset.get_scene_target(scene) <= 0.75 * dataset.target_voxel_size)).cuda()
        for metric in metrics:
            metric(nn1, target)
    return [m.compute().cpu().item() for m in metrics]


class RetrievalInterface:

    def __init__(self, config_query, latent_dim):
        self.config = config_query
        self.latent_dim = latent_dim

    def get_retrieval_mapping(self, fenc, extraction_func, tree_path, dataset, ignore_patches_from_source):
        patch_names, feats_input = extraction_func(fenc, self.config, self.latent_dim, dataset)
        retrieval_mapping = query_dictionary_using_features(self.config, patch_names, feats_input, dataset, tree_path, ignore_patches_from_source)
        return retrieval_mapping

    def get_features(self, fenc_input, fenc_target, dataset):
        patch_names_0, feats_input = extract_input_features(fenc_input, self.config, self.latent_dim, dataset)
        patch_names_1, feats_target = extract_target_features(fenc_target, self.config, self.latent_dim, dataset)
        assert len(patch_names_0) == len(patch_names_1) and sorted(patch_names_0) == sorted(patch_names_1)
        return patch_names_0, feats_input, feats_target

    @staticmethod
    def retrieve_nearest_scenes(retrieval_mapping, scene, K, tree_path, dataset_train, dataset):
        return create_retrieval_from_mapping(scene, retrieval_mapping, K, dataset_train, dataset, tree_path)

    @staticmethod
    def retrieve_nearest_scenes_for_all(retrieval_mapping, scenes, K, tree_path, dataset_train, dataset):
        retrieved_scenes = []
        for scene in tqdm(scenes, desc='recompose_scenes'):
            retrieved_scenes.append(RetrievalInterface.retrieve_nearest_scenes(retrieval_mapping, scene, K, tree_path, dataset_train, dataset).unsqueeze(0))
        return torch.cat(retrieved_scenes, dim=0)

    def create_mapping_and_retrieve_nearest_scenes_for_all(self, fenc_input, tree_path, dataset_train, dataset, K, ignore_patches_from_source):
        return RetrievalInterface.retrieve_nearest_scenes_for_all(self.get_retrieval_mapping(fenc_input, extract_input_features, tree_path, dataset, ignore_patches_from_source), dataset.scenes, K, tree_path, dataset_train, dataset)


def retrievals_to_disk(mode, config, use_target_for_feats, num_proc=1, proc=0):
    ckpt_experiment = Path(config['retrieval_ckpt']).parents[0].name
    ckpt_epoch = Path(config['retrieval_ckpt']).name.split('.')[0]
    task_dir = f"{config['task']}_{config['dataset_train']['num_points']:04d}"
    retrievals_dir = get_retrievals_dir(config)
    tree_path = Path("runs", 'retrieval_scratch', task_dir, config['dataset_train']['dataset_name'], config['dataset_train']['splits_dir'], ckpt_experiment, ckpt_epoch, str(config['K']))

    scene_handler_train = SceneHandler('train', config)
    scene_handler_val = SceneHandler('val', config)
    dataset_train = PatchedSceneDataset('train', config['dataset_train'], scene_handler_train)
    dataset_val = PatchedSceneDataset('val', config['dataset_val'], scene_handler_val)

    if mode == 'map':
        fenc_input, fenc_target = get_retrieval_networks(model_config=config['retrieval_model'])
        fenc_input = load_net_and_set_to_eval(fenc_input, Path(config['retrieval_ckpt']), "fenc_input")
        fenc_target = load_net_and_set_to_eval(fenc_target, Path(config['retrieval_ckpt']), "fenc_target")
        retrievals_dir.mkdir(exist_ok=True, parents=True)
        create_dictionary(fenc_target, config['dictionary'], config['retrieval_model']['latent_dim'], dataset_train, tree_path)
        retrieval_handler = RetrievalInterface(config["query"], config['retrieval_model']['latent_dim'])

        fenc = fenc_input if not use_target_for_feats else fenc_target
        extract_feats = extract_input_features if not use_target_for_feats else extract_target_features

        retrieval_mappings = retrieval_handler.get_retrieval_mapping(fenc, extract_feats, tree_path, dataset_train, True)
        with Timer('np_save_train'):
            np.save(retrievals_dir / "map_train.npy", retrieval_mappings)
        retrieval_mappings = retrieval_handler.get_retrieval_mapping(fenc, extract_feats, tree_path, dataset_val, False)
        with Timer('np_save_val'):
            np.save(retrievals_dir / "map_val.npy", retrieval_mappings)
    elif mode == "compose":
        (retrievals_dir / "compose").mkdir(exist_ok=True, parents=True)
        map_name = ['map_train.npy', 'map_val.npy']
        datasets = [dataset_train, dataset_val]
        for d_idx, dataset in enumerate(datasets):
            split_shapes = [x for i, x in enumerate(dataset.scenes) if i % num_proc == proc]
            retrieval_mapping = np.load(retrievals_dir / map_name[d_idx], allow_pickle=True)[()]
            for scene in tqdm(split_shapes, desc=f'recompose_scenes_{["train", "val"][d_idx]}'):
                retrieval = RetrievalInterface.retrieve_nearest_scenes(retrieval_mapping, scene, config['K'], tree_path, dataset_train, dataset)
                np.savez_compressed(retrievals_dir / "compose" / f"{scene}.npz", retrieval.numpy())
    elif mode == "evaluate":
        retrievals = []
        for scene in tqdm(dataset_val.scenes, desc='evaluate'):
            retrieval = np.load(retrievals_dir / 'compose' / f'{scene}.npz')["arr_0"]
            retrievals.append(retrieval[:1, :, :, :])
        print(get_metrics_for_retrieval(torch.from_numpy(np.stack(retrievals, axis=0)), dataset_val))


if __name__ == '__main__':
    """
        script to create retrievals
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--retrieval_ckpt', type=str, default=None)
    parser.add_argument('--mode', type=str, nargs='+')
    parser.add_argument('--proc', type=int, default=0, help='process id')
    parser.add_argument('--K', type=int, default=4, help='kNN')
    parser.add_argument('--num_proc', type=int, default=1, help='num processes')
    parser.add_argument('--no_preload', action='store_true')
    parser.add_argument('--target_query', action='store_true')

    args = parser.parse_args()
    _config = config_handler.read_config(args.config, args)
    _config['query']['K'] = _config['K']
    if args.no_preload:
        _config['dataset_train']['preload_scenes'] = False
        _config['dataset_val']['preload_scenes'] = False
    for mode in args.mode:
        retrievals_to_disk(mode, _config, args.target_query, args.num_proc, args.proc)
