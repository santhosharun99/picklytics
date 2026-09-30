#!/bin/zsh
# Train ball then court detector, detached from the terminal session:
#   nohup training/run_all.sh > logs/run_all.log 2>&1 &
cd "$(dirname "$0")/.."
if [[ ! -f models/ball_detector.pt ]]; then
  if [[ -f runs/ball/weights/last.pt ]]; then
    python3 -c "from ultralytics import YOLO; YOLO('runs/ball/weights/last.pt').train(resume=True)" >> logs/train_ball.log 2>&1
    cp runs/ball/weights/best.pt models/ball_detector.pt
  else
    python3 training/train_ball.py >> logs/train_ball.log 2>&1
  fi
fi
[[ -f models/court_detector.pt ]] || python3 training/train_court.py >> logs/train_court.log 2>&1
echo "all done $(date)"
