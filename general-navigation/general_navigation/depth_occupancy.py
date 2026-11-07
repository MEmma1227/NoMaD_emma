"""RGB-D depth -> local occupancy grid for the NoMaD safety filter.

Replaces the Depth Anything V2 stage of the proposal with the simulator's own
depth camera (``sensor.camera.depth``), mounted at exactly the same pose,
resolution and FOV as the RGB camera NoMaD sees.  Everything below the CARLA
glue (``DepthCamera``) is pure numpy so it can be unit-tested without CARLA.

Frames used in this module
--------------------------
* CARLA / UE camera frame : x forward, y right, z up (left-handed).
* Optical frame (proposal): X_c right, Y_c down, Z_c forward (= depth).
* Robot frame             : NoMaD local frame, x forward, y LEFT, z up, with
                            z measured from the bottom of the ego bounding box
                            (i.e. height above a flat road).  NoMaD candidate
                            trajectories are expressed in this same x/y frame.
"""
import math
import threading
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Decoding and geometry
# --------------------------------------------------------------------------

DEPTH_FAR_METERS = 1000.0  # CARLA depth encoding range.


def decode_carla_depth(raw_bgra):
    """CARLA depth image (H, W, 4) uint8 BGRA -> planar depth in meters.

    CARLA stores the distance along the camera's forward axis (z-buffer style
    planar depth, not Euclidean range) as R + G*256 + B*256^2 over 24 bits.
    """
    a = np.asarray(raw_bgra)
    b = a[..., 0].astype(np.float64)
    g = a[..., 1].astype(np.float64)
    r = a[..., 2].astype(np.float64)
    normalized = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
    return (normalized * DEPTH_FAR_METERS).astype(np.float32)


def intrinsics_from_fov(width, height, fov_deg):
    """Pinhole intrinsics of a CARLA camera (square pixels, centered)."""
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return dict(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
                width=int(width), height=int(height), fov_deg=float(fov_deg))


def ue_matrix(x, y, z, pitch, yaw, roll):
    """4x4 matrix of a CARLA Transform (same formula as carla.Transform.get_matrix)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0]])


# Optical (right, down, forward) -> UE camera (forward, right, up).
OPTICAL_TO_UE = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])
# UE vehicle (forward, right, up) -> robot / NoMaD (forward, left, up).
UE_TO_ROBOT = np.diag([1.0, -1.0, 1.0])


def camera_extrinsics(mount_xyz, mount_pyr, ground_offset_z):
    """Return (R, t) mapping optical-frame points into the robot frame.

    mount_xyz / mount_pyr : camera attachment relative to the vehicle, as in
        the carla.Transform used to attach it (meters, degrees).
    ground_offset_z : z of the bottom of the ego bounding box in the vehicle
        frame (bbox.location.z - bbox.extent.z); subtracted so robot z is the
        height above the road.
    """
    m = ue_matrix(*mount_xyz, *mount_pyr)
    R = UE_TO_ROBOT @ m[:3, :3] @ OPTICAL_TO_UE
    t = UE_TO_ROBOT @ m[:3, 3] - np.array([0.0, 0.0, ground_offset_z])
    return R, t

# 深度圖 → 相機座標的 3D 點 $P_c$
def back_project(depth, K, max_range=None, stride=1):
    """Depth (H, W) -> optical-frame points P_c = [X_c, Y_c, Z_c] (N, 3)."""
    d = np.asarray(depth, dtype=np.float64)[::stride, ::stride]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * stride + 0.5  # center of the sampled pixel
    v = v * stride + 0.5
    valid = np.isfinite(d) & (d > 0)
    if max_range is not None:
        valid &= d <= max_range
    z = d[valid]
    x = (u[valid] - K['cx']) * z / K['fx']
    y = (v[valid] - K['cy']) * z / K['fy']
    return np.column_stack((x, y, z))


# --------------------------------------------------------------------------
# Occupancy grid
# --------------------------------------------------------------------------

@dataclass
class GridConfig:
    resolution: float = 0.1     # meters per cell
    x_min: float = -2.0         # robot frame bounds (x forward, y left)
    x_max: float = 20.0
    y_min: float = -8.0
    y_max: float = 8.0
    h_min: float = 0.2          # keep points with h_min < z < h_max
    h_max: float = 2.5
    max_range: float = 25.0     # ignore depth beyond this (incl. sky = 1000 m)
    stride: int = 1             # pixel subsampling

    @property
    def shape(self):
        return (int(math.ceil((self.x_max - self.x_min) / self.resolution)),
                int(math.ceil((self.y_max - self.y_min) / self.resolution)))


@dataclass
class OccupancyGrid:
    occupied: np.ndarray        # (nx, ny) bool, index i along x, j along y
    observed: np.ndarray        # (nx, ny) bool, any depth return fell here
    config: GridConfig
    frame: int = -1
    pose: tuple = None          # vehicle (x, y, yaw) in world at capture time

    def to_index(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        c = self.config
        i = np.floor((xy[:, 0] - c.x_min) / c.resolution).astype(int)
        j = np.floor((xy[:, 1] - c.y_min) / c.resolution).astype(int)
        return i, j

    def cell_centers(self, mask):
        c = self.config
        i, j = np.nonzero(mask)
        return np.column_stack((c.x_min + (i + 0.5) * c.resolution,
                                c.y_min + (j + 0.5) * c.resolution))

    def inflated(self, radius):
        """Occupied cells dilated by a disk of ``radius`` meters."""
        cells = int(math.ceil(radius / self.config.resolution))
        if cells <= 0:
            return self.occupied.copy()
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells + 1, 2 * cells + 1))
        return cv2.dilate(self.occupied.astype(np.uint8), k).astype(bool)

    def distance_field(self):
        """Meters from each cell center to the nearest occupied cell (cached)."""
        if getattr(self, '_dist', None) is None:
            free = (~self.occupied).astype(np.uint8)
            if free.all():
                self._dist = np.full(self.occupied.shape, np.inf)
            else:
                self._dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.config.resolution
        return self._dist

    def path_clearance(self, path_local, inflated=None, step=None):
        """(collides, min_clearance_m) for a robot-frame path (N, 2).

        The path (prefixed with the robot origin) is densified to half a cell.
        Points outside the grid are treated as unknown and ignored, matching a
        local, forward-looking grid. ``collides`` tests ``inflated`` (or the raw
        occupancy when None); clearance is centerline-to-nearest-occupied.
        """
        path = np.asarray(path_local, dtype=float).reshape(-1, 2)
        if not len(path):
            return False, float('inf')
        path = np.vstack(([0.0, 0.0], path))
        step = step or self.config.resolution / 2.0
        dense = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
            dense.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        i, j = self.to_index(np.asarray(dense))
        mask = self.occupied if inflated is None else inflated
        inside = (i >= 0) & (i < mask.shape[0]) & (j >= 0) & (j < mask.shape[1])
        if not inside.any():
            return False, float('inf')
        i, j = i[inside], j[inside]
        return bool(mask[i, j].any()), float(self.distance_field()[i, j].min())

    def front_clearance(self, front_x, half_width):
        """Free distance ahead of the front bumper inside the ego corridor."""
        pts = self.cell_centers(self.occupied)
        if not len(pts):
            return float('inf')
        ahead = pts[(pts[:, 0] > front_x) & (np.abs(pts[:, 1]) <= half_width)]
        if not len(ahead):
            return float('inf')
        return float(ahead[:, 0].min() - front_x)
    
def extend_path(path, horizon):
    """Extend a path straight along its last segment until its length is ``horizon`` m.
 
    Used only for collision checking: NoMaD's 8 waypoints span ~2 m, too short
    to see an obstacle a few meters ahead. Paths already that long, or
    ``horizon <= 0``, are returned unchanged.
    """
    p = np.asarray(path, dtype=float).reshape(-1, 2)
    if horizon <= 0 or not len(p):
        return p
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) + float(np.linalg.norm(p[0]))
    if length >= horizon:
        return p
    d = p[-1] - p[-2] if len(p) > 1 else p[-1]
    n = np.linalg.norm(d)
    d = np.array([1.0, 0.0]) if n < 1e-6 else d / n
    return np.vstack((p, p[-1] + d * (horizon - length)))

def build_occupancy(depth, K, R, t, config: GridConfig, frame=-1, pose=None):
    """Proposal Sec. 3: D(u,v) -> P_c -> P_r = R P_c + t -> height filter -> grid."""
    pc = back_project(depth, K, config.max_range, config.stride)
    pr = pc @ R.T + t
    nx, ny = config.shape
    i = np.floor((pr[:, 0] - config.x_min) / config.resolution).astype(int)
    j = np.floor((pr[:, 1] - config.y_min) / config.resolution).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    observed = np.zeros((nx, ny), dtype=bool)
    observed[i[inside], j[inside]] = True
    obstacle = inside & (pr[:, 2] > config.h_min) & (pr[:, 2] < config.h_max)
    occupied = np.zeros((nx, ny), dtype=bool)
    occupied[i[obstacle], j[obstacle]] = True
    return OccupancyGrid(occupied=occupied, observed=observed, config=config,
                         frame=frame, pose=pose)


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
# 把深度圖（單位是公尺）轉成彩色影像，放進影片和串流的中間那一格
def colorize_depth(depth, max_depth=30.0):
    d = np.clip(np.nan_to_num(depth, nan=max_depth), 0, max_depth) / max_depth
    return cv2.applyColorMap((255 * (1.0 - d)).astype(np.uint8), cv2.COLORMAP_TURBO)


def render_bev(grid, out_hw, candidates=None, selected=None, feasible=None,
               ego_box=None, inflate_radius=None):
    """Bird's-eye view (BGR). x forward = up, y left = left."""
    h, w = out_hw
    if grid is None:
        img = np.zeros((h, w, 3), np.uint8)
        cv2.putText(img, 'no depth', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .35, (180, 180, 180), 1)
        return img
    nx, ny = grid.occupied.shape
    canvas = np.zeros((nx, ny, 3), np.uint8)
    canvas[grid.observed] = (60, 60, 60)
    if inflate_radius:
        canvas[grid.inflated(inflate_radius) & ~grid.occupied] = (0, 0, 140)
    canvas[grid.occupied] = (255, 255, 255)
    # Row = x (flip so forward is up), column = y (flip so left is left).
    canvas = canvas[::-1, ::-1]
    scale = min(h / nx, w / ny)
    img = cv2.resize(canvas, (max(1, int(ny * scale)), max(1, int(nx * scale))),
                     interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((h, w, 3), np.uint8)
    oy, ox = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    pad[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
    c = grid.config

    def px(points):
        p = np.asarray(points, float).reshape(-1, 2)
        col = (c.y_max - p[:, 1]) / c.resolution * scale + ox
        row = (c.x_max - p[:, 0]) / c.resolution * scale + oy
        return np.rint(np.column_stack((col, row))).astype(np.int32)

    if ego_box is not None:
        rear, front, half = ego_box
        cv2.rectangle(pad, tuple(px([front, half])[0]), tuple(px([rear, -half])[0]), (0, 200, 0), 1)
    if candidates is not None:
        for k, path in enumerate(candidates):
            ok = True if feasible is None else feasible[k]
            pts = px(np.vstack(([0, 0], np.asarray(path, float))))
            cv2.polylines(pad, [pts], False, (0, 255, 255) if ok else (0, 0, 255), 1, cv2.LINE_AA)
    if selected is not None:
        pts = px(np.vstack(([0, 0], np.asarray(selected, float))))
        cv2.polylines(pad, [pts], False, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(pad, 'BEV occ', (2, 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 1)
    return pad


# --------------------------------------------------------------------------
# CARLA glue
# --------------------------------------------------------------------------

class DepthCamera:
    """A depth camera rigidly co-located with the RGB camera NoMaD uses.

    CameraManager only keeps ONE sensor alive (switching to its depth entry
    would destroy the RGB stream NoMaD needs), so this spawns an independent
    ``sensor.camera.depth`` with the RGB sensor's mount, size and FOV.
    """

    def __init__(self, client, keep_frames=32):
        import carla  # local import keeps the numpy part CARLA-free
        cam = client.world.camera_manager
        rgb = cam.sensor
        attrs = rgb.attributes
        width, height = int(attrs['image_size_x']), int(attrs['image_size_y'])
        fov = float(attrs.get('fov', 90.0))
        entry = cam._camera_transforms[cam.transform_index]
        mount = entry[0]
        attachment = entry[1] if len(entry) > 1 else carla.AttachmentType.Rigid
        bp = client.sim_world.get_blueprint_library().find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        bp.set_attribute('fov', str(fov))
        if bp.has_attribute('sensor_tick') and 'sensor_tick' in attrs:
            bp.set_attribute('sensor_tick', attrs['sensor_tick'])
        player = client.world.player
        self.sensor = client.sim_world.spawn_actor(
            bp, mount, attach_to=player, attachment_type=attachment)
        box = player.bounding_box
        self.ground_offset_z = box.location.z - box.extent.z
        # Ego footprint in the robot frame: (rear_x, front_x, half_width).
        self.ego_box = (box.location.x - box.extent.x, box.location.x + box.extent.x,
                        box.extent.y)
        self.K = intrinsics_from_fov(width, height, fov)
        loc, rot = mount.location, mount.rotation
        self.mount = dict(xyz=[loc.x, loc.y, loc.z], pyr=[rot.pitch, rot.yaw, rot.roll])
        self.R, self.t = camera_extrinsics(self.mount['xyz'], self.mount['pyr'],
                                           self.ground_offset_z)
        self._lock = threading.Lock()
        self._frames = {}
        self._keep = keep_frames
        self.sensor.listen(self._on_image)

    def _on_image(self, image):
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        depth = decode_carla_depth(raw)
        with self._lock:
            self._frames[image.frame] = depth
            for f in [f for f in self._frames if f < image.frame - self._keep]:
                del self._frames[f]

    def latest(self, frames_with_pose):
        """Newest (frame, depth) whose vehicle pose is known, else None."""
        with self._lock:
            usable = [f for f in self._frames if f in frames_with_pose]
            if not usable:
                return None
            f = max(usable)
            return f, self._frames[f]

    def metadata(self, config: GridConfig):
        return dict(sensor='sensor.camera.depth (simulator ground truth, replaces Depth Anything V2)',
                    intrinsics=self.K, mount=self.mount,
                    ground_offset_z=self.ground_offset_z, ego_box=self.ego_box,
                    R_optical_to_robot=self.R.tolist(), t_optical_to_robot=self.t.tolist(),
                    grid=asdict(config))

    def destroy(self):
        try:
            self.sensor.stop()
            self.sensor.destroy()
        except RuntimeError:
            pass
