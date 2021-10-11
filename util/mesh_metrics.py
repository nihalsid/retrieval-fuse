import multiprocessing
from collections import defaultdict
from pathlib import Path
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
import shutil
import numpy as np

from util.intersections import slice_mesh_plane


def compute_iou(mesh_pred, mesh_target):
    res = 1.1875
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)

    v_pred_filled = set(tuple(x) for x in v_pred.points)
    v_target_filled = set(tuple(x) for x in v_target.points)
    iou = len(v_pred_filled.intersection(v_target_filled)) / len(v_pred_filled.union(v_target_filled))
    return iou


def compute_metrics(path_pred, path_target):
    mesh_pred = trimesh.load_mesh(path_pred)
    mesh_target = trimesh.load_mesh(path_target)
    iou = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(100000, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(100000, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.linspace(64. / 1000, 64, 1000)

    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy ** 2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]

    return [iou, chamferL1, normals_correctness, F[9], F[14]]


def compute_metrics_only_iou(path_pred, path_target):
    mesh_pred = trimesh.load_mesh(path_pred)
    mesh_target = trimesh.load_mesh(path_target)
    iou = compute_iou(mesh_pred, mesh_target)
    return [iou]


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """ Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def compute_all_metrics_for_scenes(dataset, task, method_name, base_path, scene_chunk_dict, num_proc, proc, limit=None):
    scenes = sorted(list(x.name.split(".")[0] for x in base_path.iterdir()))[:limit] #sorted(list(scene_chunk_dict.keys()))[:limit]
    worker_items = [x for i, x in enumerate(scenes) if i % num_proc == proc]
    result_list = []
    for s in tqdm(worker_items):
        try:
            retval = compute_all_metrics_for_scene(base_path, s, 1)
            result_list.append(retval)
        except Exception as e:
            print("Exception for", s, ":", e)
    print("Items recieved:", len(result_list))
    Path(f"metrics_{dataset}_{task}_{method_name}_{proc:02d}.csv").write_text("\n".join([",".join([str(x) for x in list_item]) for list_item in result_list]))


def compute_all_metrics_for_scene(base_path, scene, num_chunks):
    path_to_target = base_path.parents[0] / "gt" / (scene + ".obj")
    path_to_ours = base_path / (scene + ".obj")
    metrics = compute_metrics(path_to_ours, path_to_target)
    # print([scene] + metrics + [num_chunks])
    return [scene] + metrics + [num_chunks]


def convert_ifnet(base_dir, target_dir, samples, limit=None):
    target_dir.mkdir(exist_ok=True)
    for s in tqdm(samples[:limit]):
        mesh = trimesh.load(base_dir / s / "surface_reconstruction.off")
        mesh.export(target_dir / (s + ".obj"))


def convert_spsr(base_dir, target_dir, samples, limit=None):
    target_dir.mkdir(exist_ok=True)
    for s in tqdm(samples[:limit]):
        try:
            mesh = trimesh.load(base_dir / s, force='mesh')
            mesh.apply_scale(64)
            mesh.apply_translation(np.array([32, 32, 32]))
            mesh.export(target_dir / (s.split('.')[0] + ".obj"))
        except Exception as err:
            print(s, err)


def rescale_conv_occ(base_dir, target_dir, samples, limit=None):
    target_dir.mkdir(exist_ok=True)
    for s in tqdm(samples[:limit]):
        mesh = trimesh.load(base_dir / (s + ".off"))
        mesh.apply_scale(64)
        mesh.apply_translation([32, 32, 32])
        mesh.export(target_dir / (s + ".obj"))


def rescale_parallel(func_name, base_dir, target_dir, samples, limit=None):
    num_processes = 8
    items_per_worker = len(samples[:limit]) // num_processes + 1
    process = []
    for pid in range(num_processes):
        worker_items = samples[:limit][pid * items_per_worker: (pid + 1) * items_per_worker]
        process.append(multiprocessing.Process(target=func_name, args=(base_dir, target_dir, worker_items)))
    for p in process:
        p.start()
    for p in process:
        p.join()


def copy_scenes_for_visual_inspection(target_scenes_dir, all_methods, samples):
    outdir = Path("inspect")
    outdir.mkdir(exist_ok=True)
    for s in tqdm(samples):
        for x in all_methods:
            if (target_scenes_dir / f"{x}" / (s + ".obj")).exists():
                shutil.copyfile(target_scenes_dir / f"{x}" / (s + ".obj"), outdir / (s + f"_{x}.obj"))
            else:
                print("NotFound:", (target_scenes_dir / f"{x}" / (s + ".obj")))


def recompose_scene(base_path, chunks, suffix, shift):
    xyz = []
    meshes = []
    for chunk in chunks:
        try:
            meshes.append(trimesh.load(base_path / (chunk + suffix), force='mesh'))
            xyz.append([int(y) for y in chunk.split("__")[-1].split("_")])
        except Exception as e:
            print("Exception load_mesh: ", e)
    non_empty_meshes = [type(x) == trimesh.Trimesh for x in meshes]
    xyz = np.array(xyz)
    # joining_shift = [np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])]
    for i in range(len(meshes)):
        if non_empty_meshes[i]:
            meshes[i].apply_translation(xyz[i, :])
            # for j in range(3):
            #     if xyz[i, j] != 0:
            #         meshes[i].apply_translation(joining_shift[j] * (xyz[i, j] // 64))
    if np.array(non_empty_meshes).any():
        try:
            meshes = [m for m in meshes if type(m) == trimesh.Trimesh]
            concat_mesh = trimesh.util.concatenate(meshes)
            concat_mesh.apply_translation(shift)
            return concat_mesh
        except Exception as e:
            return None
    else:
        return None


def recompose_chunks_to_scenes(base_path, suffix, output_path, shift):
    output_path.mkdir(exist_ok=True)
    scenes_chunk_dict = get_scenes_chunk_dict(base_path, suffix)
    for scene in tqdm(sorted(scenes_chunk_dict.keys())):
        rescene = recompose_scene(base_path, scenes_chunk_dict[scene], suffix, shift)
        if rescene is not None:
            rescene.export(output_path / (scene + ".obj"))


def get_scenes_chunk_dict(base_path, suffix):
    all_chunks = [(x.name.split(suffix)[0], "__".join(x.name.split(suffix)[0].split("__")[:2])) for x in base_path.iterdir() if x.name.endswith(suffix)]
    scenes_chunk_dict = defaultdict(list)
    for chunk in all_chunks:
        scenes_chunk_dict[chunk[1]].append(chunk[0])
    return scenes_chunk_dict


def copy_crop_psr(all_samples, target_dir):
    target_dir.mkdir(exist_ok=True)
    for s in tqdm(all_samples):
        mesh = trimesh.load(s, force='mesh')
        bbox = mesh.bounding_box.bounds
        # scaled_extents = np.array([(bbox[1] - bbox[0])[0] * 2, bbox[1][1] - 4, (bbox[1] - bbox[0])[2] * 2])
        scaled_extents = np.array([(bbox[1] - bbox[0])[0] * 2, 64 - 4, (bbox[1] - bbox[0])[2] * 2])
        box = trimesh.creation.box(extents=scaled_extents)
        box.apply_translation(scaled_extents / 2)
        mesh = slice_mesh_plane(mesh=mesh, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)
        mesh.export(target_dir / f"{s.name.split('___poisson.ply')[0]}.obj")


def copy_ours(all_samples, suffix, target_dir):
    target_dir.mkdir(exist_ok=True)
    for s in tqdm(all_samples):
        shutil.copyfile(s, target_dir / f"{s.name.split(suffix)[0]}.obj")


def clean_mesh(target_dir):
    (target_dir.parents[0] / (target_dir.name + "_clean")).mkdir(exist_ok=True)
    for x in tqdm(list(target_dir.iterdir())):
        mesh = trimesh.load(x, force='mesh')
        extents = np.array([62, 62, 62])
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(np.array([64, 64, 64]) / 2)
        mesh = slice_mesh_plane(mesh=mesh, plane_normal=-box.facets_normal, plane_origin=box.facets_origin)
        mesh.export(target_dir.parents[0] / (target_dir.name + "_clean") / x.name)
