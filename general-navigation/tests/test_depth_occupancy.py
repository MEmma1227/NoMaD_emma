"""Synthetic checks for depth_occupancy (no CARLA needed).

A pinhole camera is ray-cast against a flat road (z = 0) and a box obstacle
given in the robot frame; the planar depth is encoded exactly like CARLA's
sensor.camera.depth and fed through the real decoding/projection pipeline.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[1] / 'general_navigation' / 'depth_occupancy.py'
_spec = importlib.util.spec_from_file_location('depth_occupancy', _path)
do = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(do)

W, H, FOV = 256, 128, 90.0
BOX = dict(x=(10.0, 12.0), y=(-1.0, 1.0), z=(0.0, 1.5))


def encode_carla_depth(depth_m):
    n = np.clip(depth_m / do.DEPTH_FAR_METERS, 0, 1) * (256 ** 3 - 1)
    n = np.rint(n).astype(np.int64)
    r, g, b = n % 256, (n // 256) % 256, n // 65536
    return np.stack([b, g, r, np.full_like(r, 255)], -1).astype(np.uint8)


def render(R, t, K, box=BOX):
    v, u = np.mgrid[0:H, 0:W] + 0.5
    rays_opt = np.stack(((u - K['cx']) / K['fx'], (v - K['cy']) / K['fy'], np.ones_like(u)), -1)
    dirs = rays_opt @ R.T  # robot frame, scaled so optical z == 1 => s == planar depth
    best = np.full((H, W), do.DEPTH_FAR_METERS)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -t[2] / dirs[..., 2]  # ground plane
        best = np.where((s > 0) & np.isfinite(s), np.minimum(best, s), best)
        lo = np.full((H, W), -np.inf); hi = np.full((H, W), np.inf)
        for k, (a, b) in enumerate((box['x'], box['y'], box['z'])):
            s1, s2 = (a - t[k]) / dirs[..., k], (b - t[k]) / dirs[..., k]
            lo = np.maximum(lo, np.minimum(s1, s2)); hi = np.minimum(hi, np.maximum(s1, s2))
        hit = (hi >= lo) & (lo > 0)
        best = np.where(hit, np.minimum(best, lo), best)
    return best


def setup(pitch=0.0):
    K = do.intrinsics_from_fov(W, H, FOV)
    R, t = do.camera_extrinsics([1.5, 0.0, 1.6], [pitch, 0.0, 0.0], ground_offset_z=0.0)
    depth = do.decode_carla_depth(encode_carla_depth(render(R, t, K)))
    return K, R, t, depth


def test_axes():
    R, t = do.camera_extrinsics([1.0, 0.5, 2.0], [0, 0, 0], ground_offset_z=-0.3)
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-12)   # forward
    np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-12)  # image right -> robot -y
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-12)  # image down -> robot -z
    np.testing.assert_allclose(t, [1.0, -0.5, 2.3])                   # CARLA y right -> robot y left
    R, _ = do.camera_extrinsics([0, 0, 0], [0, 90, 0], 0)
    np.testing.assert_allclose(R @ [0, 0, 1], [0, -1, 0], atol=1e-12)  # yaw +90 looks right
    R, _ = do.camera_extrinsics([0, 0, 0], [-10, 0, 0], 0)
    assert (R @ [0, 0, 1])[2] < 0                                     # negative pitch looks down


def test_decode_roundtrip():
    d = np.array([[0.5, 7.25, 30.0, 999.0]])
    np.testing.assert_allclose(do.decode_carla_depth(encode_carla_depth(d)), d, atol=1e-3)


def check_scene(pitch):
    K, R, t, depth = setup(pitch)
    cfg = do.GridConfig()
    grid = do.build_occupancy(depth, K, R, t, cfg)
    pts = grid.cell_centers(grid.occupied)
    assert len(pts) > 20, 'box should be seen'
    # Every occupied cell lies on the box footprint (only its front face is visible).
    r = cfg.resolution
    assert pts[:, 0].min() >= BOX['x'][0] - r and pts[:, 0].max() <= BOX['x'][1] + r
    assert pts[:, 1].min() >= BOX['y'][0] - r and pts[:, 1].max() <= BOX['y'][1] + r
    assert abs(pts[:, 0].min() - BOX['x'][0]) <= r, 'front face position'
    # Ground was observed but never marked occupied.
    assert grid.observed.sum() > 10 * grid.occupied.sum()
    front_x = 2.3
    assert abs(grid.front_clearance(front_x, 1.0) - (BOX['x'][0] - front_x)) <= r
    straight = np.column_stack((np.linspace(0.5, 15, 8), np.zeros(8)))
    side = np.column_stack((np.linspace(0.5, 15, 8), np.full(8, 3.5)))
    inflated = grid.inflated(1.2)
    hit, c_hit = grid.path_clearance(straight, inflated)
    ok, c_ok = grid.path_clearance(side, inflated)
    assert hit and c_hit < r
    assert not ok and 2.3 < c_ok < 2.7
    return grid


def test_scene_level():
    check_scene(0.0)


def test_scene_pitched_down():
    check_scene(-8.0)


def test_height_filter_drops_overhead():
    K = do.intrinsics_from_fov(W, H, FOV)
    R, t = do.camera_extrinsics([1.5, 0.0, 1.6], [0, 0, 0], 0.0)
    overhead = dict(x=(8.0, 9.0), y=(-2.0, 2.0), z=(3.0, 3.5))  # above h_max
    grid = do.build_occupancy(render(R, t, K, overhead), K, R, t, do.GridConfig())
    assert not grid.occupied.any()


def test_render_bev_shape():
    K, R, t, depth = setup()
    grid = do.build_occupancy(depth, K, R, t, do.GridConfig())
    img = do.render_bev(grid, (128, 128), candidates=[[[5, 0], [10, 0]]], feasible=[False],
                        selected=[[5, 2], [10, 3]], ego_box=(-2.3, 2.3, 1.0), inflate_radius=1.2)
    assert img.shape == (128, 128, 3)
    assert do.render_bev(None, (128, 128)).shape == (128, 128, 3)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
