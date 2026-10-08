# Ball tracking & rally physics - research plan (started 2026-10-05)

Goal: a publishable, honestly evaluated system for pickleball broadcast video: ball detection/tracking, 3D trajectory,
contact / bounce / net events, rally ending (net / out / in) and line calls - with every number measured on held-out,
human-verified data and compared with baselines.

## Why the current system is not enough (measured, frame by frame, 2026-10-05)
- Single-frame YOLO11 ball detector (trained on ~69k public Roboflow images) + global path search. On TNPPL doubles
  the track follows WRONG objects for stretches (a fixed object for 19 frames mid-flight, spare balls at the frame
  edge, a ball held by a player). Earlier test: higher mAP on Roboflow images did NOT transfer to broadcast video.
- Events are separate hand-built detectors (direction changes + small classifiers trained on 2 matches):
  contacts missed / credited to the partner; end-of-point bounces found in 9 of 76 labelled rallies.
- A TrackNet gap filler (2026-10-03) trained only on our own pseudo-labels inherited their errors and was rejected.
- Nothing was ever measured against human-verified ball positions -> no reliable precision/recall.

## 1. Ground truth (needs a human labeller; ~1 h, can be split)
- 12 held-out rally clips stratified by venue / format (TNPPL indoor doubles x6 + singles x1, PPA Las Vegas doubles x2
  + singles x1, US Open singles x1, PPA Europe outer court x1), serve to point-over + 1 s.
- Every 2nd frame (15 Hz): the labeller confirms the proposed ball (Enter), clicks the true ball (with a magnifier),
  or marks 'no ball visible'. dashboard/balltruth.html -> labels/balltruth/<clip>.json.
- Whole matches held out for the cross-venue claim: TNPPL_6_MD, Las Vegas Waters/Johns, Europe feed - never used in
  training, not even for pseudo-labels.
- Metrics (TrackNet / WASB convention): precision, recall, F1 with a hit = predicted centre within 4 px (at 1280x720)
  of the true centre; 'no ball' frames count as negatives. Per venue and overall, with bootstrap 95% intervals.

## 2. Temporal heat-map ball detector (state of the art for small fast balls)
- References: TrackNet (Huang et al., IEEE AVSS 2019), TrackNetV2 (Sun et al., 2020), TrackNetV3 (Chen & Wang,
  ACM MM Asia 2023: background estimation + trajectory inpainting), WASB (Tarashima et al., BMVC 2023: HRNet,
  multi-frame in/out, strong across 5 sports). [Citations to be verified against the papers before publishing.]
- Design: input K=3 (and 5) consecutive frames + a median background frame (TrackNetV3 idea: static objects - logos,
  a ball lying by the fence - vanish in frame minus background), HRNet-W18 style backbone (ImageNet init, torchvision
  weights), outputs one heat-map per input frame; focal-style weighted BCE; online hard-negative mining.
- Training data: (a) verified frames from non-test rallies, (b) physics-consistent pseudo-labels from all 12 matches
  (only frames explained by a RANSAC 3D ballistic flight with < 3 px error; test clips / held-out matches excluded),
  (c) hard negatives: positions where the old tracker followed a non-match ball.
- Trained on Kaggle GPU (laptop overheats with heavy jobs).
- Baselines on the same test set: current YOLO11 + tracker; the 2026-10-03 TrackNet.

## 3. Physics-based 3D reconstruction (hybrid dynamical system)
- Calibrated broadcast camera (camera3d, ~2 px reprojection). The ball is a hybrid system: flight (gravity, quadratic
  air drag; spin later) / bounce (ground plane h=0, restitution) / contact (velocity jump within paddle reach of a
  player) / net (plane y=22, height < tape).
- Per rally: a dynamic programme over event sequences; each flight piece is a robust 3D ballistic fit to the
  detector's heat-map peaks (RANSAC + Levenberg-Marquardt, uncertainty from the Jacobian); event times and types are
  chosen to minimise reprojection error + event priors (alternation of sides, a bounce before the return, ...).
- Outputs with uncertainty: contacts (who: nearest player to the 3D contact point), bounces, net touches, the last live
  shot and how the rally ended, landing points with a 95% interval -> line calls 'in' / 'out' / 'too close to call'.
- Validation: event tags (US Open, Las Vegas), 90 labelled rally endings (labels/endings.json), referee logs and the
  TNPPL score overlay; label-free checks (every mid-rally shot answered by the other side must clear the net).

## 4. Reporting
- Every metric with its test set, interval and baseline in docs/RESULTS.md; the site's accuracy page reads from it.
