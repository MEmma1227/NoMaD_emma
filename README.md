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
#  虛擬環境
1. 安裝 Miniconda（已經裝過可以跳過）
```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
```
2. 建立環境
```
source ~/miniconda3/bin/activate
conda create -p ~/envs/nomad_carla python=3.10 -y
source ~/miniconda3/bin/activate ~/envs/nomad_carla
```
3. 安裝套件
順序很重要：PyTorch 要先從官方的 CUDA 版本來源安裝，否則 requirements.txt 可能會裝到不對應 GPU 的版本
```
# (1) PyTorch（對應 CUDA 12.4）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# (2) 專案本身的套件
cd ~/ECE_780_NoMaD/general-navigation
pip install -r requirements.txt
pip install -e .

# (3) requirements.txt 沒有列到，但會用到的
pip install flask
```
(4) CARLA 的 Python client：版本必須和 CARLA 模擬器完全一致
先看看模擬器是哪個版本：
```
ls <CARLA資料夾>/PythonAPI/carla/dist/
```
例如看到 carla-0.9.15-cp310-...whl，就安裝相同版本：
```
pip install carla==0.9.15
```
如果 dist/ 裡有 cp310 的 .whl 檔，也可以直接安裝那個檔案：pip install <路徑>/carla-0.9.15-cp310-....whl。

4. 確認安裝成功
```
python -c "import torch, carla, pygame, cv2, flask; print(torch.__version__, torch.cuda.is_available()); x=torch.zeros(1).cuda(); print('GPU OK')"
python tests/test_depth_occupancy.py
```
第一行應該印出 2.5.1+cu124 True 和 GPU OK；第二行應該全部顯示 ok。
