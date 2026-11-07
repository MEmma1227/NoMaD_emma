python -u -X faulthandler -m general_navigation.carla_obstacle_test \
  --checkpoint "$HOME/.cache/general_navigation/nomad.pth" \
  --headless \
  --town Town10HD_Opt \
  --reload-world \
  --spawn-index 6 \
  --goal-distance 20 \
  --goal-radius 2 \
  --obstacle-before-goal 10 \
  --check-horizon 6 \
  --rpc-timeout 120 \
  --depth-camera \
  --stream-port 8080 \
  --output "$HOME/carla_test_results/obstacle_depth"\
  # --path-selection official  #改成直接用第一條候選