CARLA重裝
```
ssh XXX@ServerName.uwaterloo.ca
Password

bash
mkdir -p /tmp/carla_ws/carla && chmod 700 /tmp/carla_ws
cd /tmp/carla_ws
wget -c -O CARLA_0.9.15.tar.gz https://downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz
tar -xf CARLA_0.9.15.tar.gz -C carla && rm CARLA_0.9.15.tar.gz
ls carla/CarlaUE4.sh
```

First, running CARLA simulator
```bash
cd /tmp/carla_ws/carla
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound 
```

Open another terminal
```
ssh -L 8080:localhost:8080 XXX@ServerName.uwaterloo.ca
Password

bash
source ~/miniconda3/bin/activate ~/envs/nomad_carla
cd ~/ECE_780_NoMaD/general-navigation
bash carla_obstacle_test.sh
```
Where,
```
# ---------------- Model & simulator ----------------
--checkpoint "$HOME/.cache/general_navigation/nomad.pth"
# Absolute path of model weight *.pth (must be absolute; never downloads when given).

--town Town10HD_Opt
# CARLA map to load.

--reload-world
# Reload the map before testing to clear prior simulation state.

--spawn-index 6
# Index of the map spawn point used as the starting point.

--carla-port 2000
# CARLA RPC port; must match -carla-rpc-port of CarlaUE4.sh (default 2000).

--rpc-timeout 120
# Timeout (s) for connecting to CARLA and loading the map.

--headless
# No pygame window; the camera video is still recorded.

# ---------------- Scenario geometry ----------------
--goal-distance 20
# Distance along the road from the starting point to the goal, in meters.

--obstacle-before-goal 10
# Distance along the road from the obstacle center to the goal, in meters.
# The obstacle is placed before the goal.
# Distance from start to obstacle = goal-distance - obstacle-before-goal.
# Must be larger than goal-radius and smaller than goal-distance.

# Obstacles: each run executes 4 trials in order:
# baseline (none), small (vehicle.harley-davidson.low_rider),
# medium (vehicle.mini.cooper_s), large (vehicle.mercedes.sprinter).

# ---------------- Success / failure criteria ----------------
--goal-radius 2
# The robot is considered to have arrived when it is within 2 meters of the goal.

--fail-distance 5
# Fail ("goal_overshoot") when the robot passes the goal line by this many meters
# along the road without having arrived.

--stuck-seconds 10
--stuck-distance 0.3
# Stuck when the robot moves less than 0.3 m within 10 simulated seconds.

--timeout-sim 0
--timeout-wall 0
# Optional time limits in simulated / wall-clock seconds; 0 disables.

# ---------------- NoMaD path selection ----------------
--path-selection current
# current : pick the candidate closest to the other candidates (consensus), with continuity.
# official: use the first sampled candidate (original NoMaD behavior).

# ---------------- RGB-D depth & occupancy grid (logging only) ----------------
--depth-camera
# Attach a ground-truth depth camera at the same pose/size/FOV as the RGB camera,
# build a robot-frame occupancy grid every step, and check NoMaD's candidates for collision.
# Control is NOT changed; results are logged to report.json and drawn in the video.

--save-depth
# Also save per-frame depth/occupancy .npz files (implies --depth-camera).

--grid-resolution 0.1
# Occupancy grid cell size (m). Grid covers x in [-2, 20] m (forward), y in [-8, 8] m (left).

--grid-h-min 0.2
--grid-h-max 2.5
# Only points with height above the road in (h_min, h_max) are obstacles.

--grid-max-range 25
# Ignore depth beyond this distance (m).

--safety-margin 0.3
# Obstacles are inflated by (robot half-width + safety-margin) m for collision checks.

--check-horizon 6
# Extend each candidate straight to this length (m) for collision checking only.
# NoMaD paths span only ~2 m; 0 = check the raw NoMaD paths.

# ---------------- Output ----------------
--output "$HOME/carla_test_results/obstacle_depth"
# Results folder. If it already has results, a new run_<timestamp> subfolder is created.

--stream-port 8080
# Serve a live MJPEG view (RGB | depth | BEV) on localhost:8080; 0 disables.
```
