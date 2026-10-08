# Results (held-out, human-verified; see docs/BALL_RESEARCH_PLAN.md)

## Ball detection - ground truth v1 (2026-10-06)
12 random rally clips, 1784 main-camera frames (every 2nd frame), every ball position clicked by hand (ball visible
in 1534, not visible in 250). Clips from tnppl_6_MD, lv_waters_johns, europe_feed_1 are from matches never used
for training. Scoring: training/eval_balltruth.py (TP = within the tolerance at 1280x720; a far prediction = FP + FN;
a prediction on a no-ball frame = FP); 95% CI by bootstrap over clips.

### Baseline: current system (YOLO11 single-frame detector + global path search, interpolation)
| tolerance | precision | recall | F1 (95% CI) |
|---|---|---|---|
| 2 px | 33.0% | 35.5% | 34.2% (30.4-38.9) |
| 4 px | 68.0% | 73.2% | 70.5% (66.9-74.3) |
| 8 px | 81.6% | 87.9% | 84.6% (79.5-89.6) |
| 16 px | 82.7% | 89.1% | 85.8% (80.4-91.1) |

At 4 px by venue: TNPPL 68.9%, PPA Las Vegas 71.9%, US Open 73.8%, PPA Europe 76.2%.
Error anatomy (1534 ball frames, 250 no-ball frames): median position error 2.4 px when on the ball; 100 frames on a
different object (> 30 px); 182 frames claiming a ball when none is visible; 64 frames with the ball missed.

### Temporal heat-map detector, round 1 (outputs/balltrack/v2; trained 6 h on Kaggle T4, 8 matches: TNPPL x7 + PPA LV
Bright/Patriquin, 16k physics-verified frames; US Open, Fahey/Yoshitomi, Europe NOT in training)
| setting | 4 px P / R / F1 | 8 px P / R / F1 |
|---|---|---|
| baseline (old system) | 68.0 / 73.2 / 70.5 | 81.6 / 87.9 / 84.6 |
| v2 thr 0.5 | 79.2 / 62.7 / 70.0 | 96.5 / 76.5 / 85.3 |
| v2 thr 0.2 | 78.3 / 67.1 / 72.3 (CI 65.9-78.1) | 95.2 / 81.6 / 87.9 (CI 81.8-92.7) |
| v2 thr 0.1 | 76.0 / 68.8 / 72.2 | 92.7 / 84.0 / 88.1 (CI 82.6-92.7) |
At 8 px, thr 0.5, by venue: TNPPL F1 90.0% (P 95.7, wrong-object false positives 37 vs 242 for the baseline);
PPA Las Vegas 90.2%; US Open 67.0% (R 50.7% - venue not in training); PPA Europe 46.4% (R 30.2% - held-out venue).
Validation F1 on held-back training rallies (pseudo-labels) was 99.3% - optimistic; only the hand labels count.
Findings: wrong-object errors largely removed; recall drops on venues absent from training; 4 px localisation limited
by the 768x432 input (60% of 1280).

### Temporal heat-map detector, round 2 (outputs/balltrack/v3; 10 training matches incl. US Open + Fahey/Yoshitomi,
25k physics-verified frames; Europe + tnppl_6_MD + Waters/Johns still never seen)
| setting | 4 px P / R / F1 (95% CI) | 8 px P / R / F1 (95% CI) |
|---|---|---|
| baseline (old system) | 68.0 / 73.2 / 70.5 (66.9-74.3) | 81.6 / 87.9 / 84.6 (79.5-89.6) |
| round 2 thr 0.5 | 78.7 / 68.8 / 73.5 (68.8-78.8) | 96.6 / 84.4 / 90.1 (84.6-95.0) |
| round 2 thr 0.2 | 76.7 / 72.0 / 74.3 (70.1-79.2) | 93.8 / 88.1 / 90.9 (86.6-94.4) |
8 px, thr 0.2, by venue: US Open 96.5% (baseline 94.5%), PPA Las Vegas 95.0% (92.4%), TNPPL 90.9% (81.5%),
PPA Europe 72.5% (80.0%; held-out venue with a much lower camera - recall 56.8%, precision 100%).
Adding the US Open / Fahey matches lifted US Open recall from 64% to 96%: venue coverage in training matters most.

### Temporal heat-map detector, round 3 (outputs/balltrack/v4; same data as round 2 at 1024x576 input, batch 3, 6 h)
(The first round-3 attempt silently trained at 768x432 - stale kernel - and died; rebuilt 2026-10-07.)
| setting | 4 px P / R / F1 (95% CI) | 8 px P / R / F1 (95% CI) |
|---|---|---|
| round 3 thr 0.5 | 80.9 / 70.6 / 75.4 (71.0-80.2) | 97.3 / 84.9 / 90.7 (85.3-94.8) |
| round 3 thr 0.2 | 78.0 / 73.7 / 75.8 (72.0-80.2) | 93.9 / 88.8 / 91.3 (87.2-95.1) |
4 px F1 by venue (thr 0.2, round 2 -> 3): US Open 69.4 -> 73.2, Las Vegas 71.0 -> 73.8, TNPPL 76.9 -> 77.8, Europe 67.0 -> 67.0.
Verdict: consistently a little better (no venue worse), but within the intervals and ~1.8x the pixels per frame ->
round 2 (models/balltrack_v3.pt) stays the default; round 3 kept in outputs/balltrack/v4 in case the physics layer
turns out to be limited by pixel accuracy. Europe (low camera) is the real gap: recall, not localisation.

## Physics layer (pickleball/rally3d.py) - US Open tagged rallies, 2026-10-08
Ball track: round-2 detector (thr 0.3). 14 tagged rallies (1 skipped: 4170-frame stretch), events matched +-3 frames.
| method | hits P / R / F1 | bounces P / R / F1 |
|---|---|---|
| current detectors (trained on these tags) | 97 / 88 / 92 | 85 / 88 / 86 |
| C: detected events + refine, linear drag 0.12 | 57 / 77 / 65 | 40 / 88 / 55 |
| C + drag 0.6 + per-flight spin (g_eff) | 73 / 82 / 77 | 43 / 88 / 57 |
| D: + live window (rally-end model) + end of play + anchored hits | 89 / 95 / 92 | 78 / 97 / 86 |
Flight model on the 79 tagged flights (true event times): pixels within 6 px 72% (linear drag 0.12) -> 94% (drag 0.6
+ per-flight effective gravity). Serves need g_eff ~47 ft/s^2 (topspin), long rally flights ~31 (gravity 32.2).

## Landing spots vs tagged bounces (US Open, 42 tagged shots; training/eval_landing.py, 2026-10-08)
Truth = tagged bounce pixel mapped through the ground homography (exact on the floor).
| method | median error | 80% within |
|---|---|---|
| physics pass (plausible shots, n 33) | 0.65 ft | 1.29 ft |
| 3D lifter (n 42) | 0.60 ft | 1.90 ft |
| 2D landing from the bounce detector (n 41; detector trained on these tags) | 0.44 ft | 2.06 ft |
Physics ~ same typical error, smaller tail. Same 37 shots: physics 0.66 vs lifter 0.53 ft median, closer on 51%.
Use: keep current landing / speed in reports; physics adds spin (g_eff), net clearance and outlier checks.
