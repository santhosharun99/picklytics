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

## Analyse a match

```bash
python3 analyze.py "path/to/match.mp4" --names P1=Devilliers,P2=Boyer          # full match
python3 analyze.py "path/to/match.mp4" --start 590 --end 650                    # a 1-minute clip
python3 analyze.py "path/to/match.mp4" --skip-detect                            # re-analyse cached detections
```

Pipeline: `pickleball/detect.py` (models → `detections.jsonl`, skips non-court camera shots) →
`pickleball/analysis.py` (players, ball track, hits/bounces, rallies, shots, stats → `analysis.json`) →
`pickleball/render.py` (annotated video with top-down court).

## Dashboard

```bash
python3 serve.py
```

Open http://127.0.0.1:8001/dashboard/ and pick a match. It shows the annotated video synced with a live
top-down court, the rally list and per-rally shot log, player stats, shot mix, shot placement,
rally lengths and court-coverage heatmaps.

## Labelling & accuracy

Hand-labelled rallies are the answer key for measuring accuracy.

1. Make a browser-friendly copy of the match (same timing, 720p):
   `ffmpeg -i match.mp4 -vf scale=1280:720 -c:v h264_videotoolbox -b:v 3000k -g 15 -c:a aac data/videos/<name>.mp4`
2. `python3 serve.py`, open http://127.0.0.1:8001/dashboard/label.html
3. Per rally: `1` = hit by near player, `2` = hit by far player, `X` = bounce, then shot keys
   (S R D P K L O T G), `F`/`B` forehand/backhand, `V` volley, click the ball for its position, `]` to end
   the rally, and pick who won the point. Labels auto-save to `labels/<name>.json`.
   Faster: `python3 training/prelabel.py outputs/<run> --video <name>` creates draft labels from a run;
   click "Import auto-labels", fix mistakes, press Enter to mark each rally reviewed (only reviewed rallies are scored).
4. Score a run: `python3 training/evaluate.py outputs/<run> labels/<name>.json`

## Datasets (CC BY 4.0)

- Pickleball Vision, Liberin Technologies: https://universe.roboflow.com/liberin-technologies/pickleball-vision
- Pickleball, pickleball-1uztf: https://universe.roboflow.com/pickleball-1uztf/pickleball-uninu
- Pickleball Court (keypoints), necromancer: https://universe.roboflow.com/necromancer/pickleball-court-vbmkq
