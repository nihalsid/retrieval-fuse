import marching_cubes as mc
import numpy as np
import trimesh
import trimesh.voxel.ops
import pyrender
from scipy.spatial.transform import Rotation
from PIL import Image


def visualize_sdf_as_voxels(sdf, output_path, level=0.5):
    from util.misc import to_point_list
    point_list = to_point_list(sdf <= level)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)


def visualize_pointcloud(pointcloud, output_path):
    f = open(output_path, "w")
    for i in range(pointcloud.shape[0]):
        x, y, z = pointcloud[i, 0], pointcloud[i, 1], pointcloud[i, 2]
        c = [1, 1, 1]
        f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()


def visualize_grid_as_voxels(grid, output_path):
    from util.misc import to_point_list
    point_list = to_point_list(grid > 0)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)


def visualize_sdf_as_mesh(sdf, output_path, level=0.75, scale_factor=1):
    vertices, triangles = mc.marching_cubes(sdf, level)
    vertices = vertices / scale_factor
    mc.export_obj(vertices, triangles, output_path)


def visualize_float_grid(grid, ignore_val, minval, maxval, output_path):
    from matplotlib import cm
    jetmap = cm.get_cmap('jet')
    norm_grid = (grid - minval) / (maxval - minval)
    f = open(output_path, "w")
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if grid[x, y, z] > ignore_val:
                    c = (np.array(jetmap(norm_grid[x, y, z])) * 255).astype(np.uint8)
                    f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()


def visualize_normals(grid, output_path):
    grid = ((grid * 0.5 + 0.5) * 255).astype(np.uint8)
    f = open(output_path, "w")
    for x in range(grid.shape[1]):
        for y in range(grid.shape[2]):
            for z in range(grid.shape[3]):
                c = grid[:, x, y, z]
                if c[0] != 127 or c[1] != 127 or c[2] != 127:
                    f.write('v %f %f %f %f %f %f\n' % (x + 0.5, y + 0.5, z + 0.5, c[0], c[1], c[2]))
    f.close()


def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def render_visualizations_to_image(mesh_dir, target_dir):
    target_dir.mkdir(exist_ok=True)
    try:
        lights = create_raymond_lights()
        r = pyrender.OffscreenRenderer(480, 480)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_rotation = np.eye(4)
        camera_rotation[:3, :3] = Rotation.from_euler('y', 0, degrees=True).as_matrix() @ Rotation.from_euler('x', -55, degrees=True).as_matrix()
        camera_translation = np.eye(4)
        camera_translation[:3, 3] = np.array([0, 0, 1.25])
        camera_pose = camera_rotation @ camera_translation
        for scene_name in list(set(['_'.join(x.name.split('_')[:-1]) for x in mesh_dir.iterdir() if x.name.endswith('.obj')])):
            images = []
            for suffix in ['_input.obj', '_pred.obj', '_gt.obj']:
                try:
                    # noinspection PyTypeChecker
                    mesh = trimesh.load(mesh_dir / (scene_name + suffix), force='mesh')
                    bbox = mesh.bounding_box.bounds
                    loc = (bbox[0] + bbox[1]) / 2
                    scale = (bbox[1] - bbox[0]).max()
                    mesh.apply_translation(-loc)
                    mesh.apply_scale(1 / scale)
                    mesh = pyrender.Mesh.from_trimesh(mesh)
                    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
                    scene.add(mesh)
                    scene.add(camera, pose=camera_pose)
                    for n in lights:
                        scene.add_node(n, scene.main_camera_node)
                    color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES)
                    images.append(color)
                # noinspection PyBroadException
                except Exception as e:
                    print('[render_visualizations_to_image]: ', e)
                    images.append(255 * np.ones((480, 480, 3)).astype(np.uint8))
            im = Image.fromarray(np.hstack(images))
            im.save(target_dir / (scene_name + '.jpg'))
            im.close()
    # noinspection PyBroadException
    except Exception as e:
        print('[render_visualizations_to_image]: ', e)
