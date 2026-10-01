#!/bin/zsh
# Train the court model first (needed for any analysis), then resume ball training.
#   nohup caffeinate -i training/run_court_then_ball.sh > logs/run_all.log 2>&1 &
cd "$(dirname "$0")/.."
[[ -f models/court_detector.pt ]] || python3 training/train_court.py --model yolo11n-pose.pt --epochs 40 >> logs/train_court.log 2>&1
echo "court done $(date)"
python3 -c "from ultralytics import YOLO; YOLO('runs/ball/weights/last.pt').train(resume=True)" >> logs/train_ball.log 2>&1
cp runs/ball/weights/best.pt models/ball_detector.pt
echo "all done $(date)"
