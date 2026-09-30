# Pickleball Analyzer

Video analysis for pickleball, like the analytics used in cricket and football:
ball tracking, player movement, shot types and match stats from game video.

## Status

| Stage | What | State |
|---|---|---|
| 1. Ball detector | YOLO11 trained on Roboflow pickleball datasets | ready to train |
| 2. Player tracking + pose | Pretrained YOLO11-pose (people + wrist/shoulder keypoints) | used in `test_detect.py` |
| 3. Court detection + top-down mapping | YOLO11-pose, 12 court keypoints → homography to a 20×44 ft court | training script ready |
| 4. Ball tracking + bounce/hit events | Trajectory smoothing, direction changes | next |
| 5. Shot classification | serve, return, dink, drive, drop, lob, volley, smash | next |
| 6. Dashboard | Video + heatmaps + shot charts + rally stats | next |

## Setup

1. Get a free Roboflow API key: roboflow.com → sign in → Settings → API Keys.
2. Paste it into `.env` after `ROBOFLOW_API_KEY=`.

## Train the ball model

```bash
python3 training/download_datasets.py   # fetch datasets (configs/datasets.yaml)
python3 training/merge_datasets.py      # unify classes -> data/merged
python3 training/train_ball.py          # train -> models/ball_detector.pt
```

Quick smoke test first: `python3 training/train_ball.py --epochs 1 --fraction 0.05`

## Train the court model

```bash
python3 training/download_datasets.py configs/court_datasets.yaml
python3 training/train_court.py         # -> models/court_detector.pt
```

Keypoint order and court coordinates are documented in `training/train_court.py`.

## Test on a video

```bash
python3 training/test_detect.py path/to/game.mp4 --max-frames 300
```

Output goes to `outputs/`.

## Datasets (CC BY 4.0)

- Pickleball Vision, Liberin Technologies: https://universe.roboflow.com/liberin-technologies/pickleball-vision
- Pickleball, pickleball-1uztf: https://universe.roboflow.com/pickleball-1uztf/pickleball-uninu
- Pickleball Court (keypoints), necromancer: https://universe.roboflow.com/necromancer/pickleball-court-vbmkq
