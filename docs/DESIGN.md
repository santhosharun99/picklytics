# How the analyzer detects and predicts everything

Plain-language map of the pipeline, how each step is checked, and a review of its logic
(written 2026-10-02, after the first test on a second venue). Accuracy numbers are from
honest tests unless marked "in-sample".

## 1. Pipeline, step by step

| # | What | How it is done | Checked how | Accuracy |
|---|---|---|---|---|
| 1 | **Court + camera** | Court keypoint model (YOLO-pose, 12 points) or a one-time hand calibration per broadcast; then *line snapping* to the white lines every frame (CourtLock). A frame is "court view" only if >=55% of the expected line pixels sit on white lines. | Lines drawn back on video; net-top prediction | Very good on known cameras. **Court model fails on a new court** (blue Las Vegas court: 0 detections) |
| 2 | **Other cameras / replays** | Frames whose court sits >40 px away from the usual (fixed) camera position are dropped. | Visual scan of the whole match | 0 bad frames let through (US Open) |
| 3 | **Ball** | YOLO ball detector v3 (12 Roboflow datasets, 960 px) + tracker that ignores still "balls" (logos) and fills short gaps. | Ball seen in % of play frames | 80-84% of play frames |
| 4 | **Players** | Pose model (17 body points) at 1280 px; one player per side of the net, closest to the court centre; feet -> court position. | Indirect (side/player of hits) | Good |
| 5 | **Who is who** | Shirt colour: 2 colour groups over the whole video + one side->player vote per camera segment. | Server vs scoreboard | 55-61/62 after the fix |
| 6 | **Hits** | Candidates (breaks in the ball's path + arm-swing peaks) -> Random Forest on clues: arm swing, ball motion, ball-to-player distance, 3D camera clues, ball-to-paddle distance. | Leave-one-rally-out on 15 tagged rallies; 3 never-seen rallies | 94% (US Open), 12/14 on unseen rallies |
| 7 | **Bounces** | Same candidates + image-y turning points -> Logistic Regression with the hit clues + where the ball meets the ground. | Leave-one-rally-out, 33 tagged bounces | ~83% (from 57%) |
| 8 | **Clean-up rules** | Hits must alternate sides (two same-side hits in a row: the earlier one becomes a bounce); at most one bounce between hits. | - | - |
| 9 | **Rallies** | Events grouped; a new rally after a >2.5 s pause or a camera cut; **needs >= 2 hits**. | vs tags / referee | 15/15 tagged rallies; **37/53 in Las Vegas** |
| 10 | **Missed serve / return** | Count the ball's net crossings before the first detected hit: each = one missed shot; the side the ball started on = server (cap 2). | vs scoreboard + tags | server 61/62, 19/19 |
| 11 | **Server** | Hitter of the first (possibly inferred) shot. | as above | 98% (US Open) |
| 12 | **Point winner** | The next rally's server (in singles the rally winner always serves next, in side-out AND rally scoring). | vs scoreboard | 58/58 (US Open) |
| 13 | **Games** | **A break > 100 s = new game**; the last rally of a game -> server won. | vs referee | 2/2 US Open, **4 instead of 2 in Las Vegas** |
| 14 | **Shot types** | Rules: position in the rally (serve/return/third), speed, flight time, landing depth, volley (no bounce before), overhead (wrist above head). | vs tags | ~65-70% (weakest) |
| 15 | **Speed** | Ground distance from hitter's feet to the bounce / flight time. | not verified | rough |
| 16 | **In / out** | Bounce ground position vs the court lines (0.5 ft margin). | vs 5 tags | weak (1/5) |
| 17 | **Movement** | Feet positions during rallies -> distance, depth, kitchen time, heatmaps. | not verified | likely within 1-2 ft |
| 18 | **Coach report** | Per-player stats + plain-language insights, each linked to the rallies behind it. | - | as good as 6-17 |

## 2. Logic review: mistakes found

Severity: **High** = gives wrong answers on real matches; **Medium** = wrong in some situations; **Low** = cosmetic or rare.

| # | Severity | Problem | Why it is wrong | Fix |
|---|---|---|---|---|
| L1 | High | **Rallies need >= 2 detected hits** | Aces, serve faults and unreturned serves have 1 hit, so these points vanish. Missing points also break step 12 (the winner of the rally before comes from the wrong next server). Las Vegas: 37 rallies found vs 53. | Keep 1-hit rallies when the hit looks like a serve (hitter behind the baseline, ball crosses the net). |
| L2 | High | **Games found by long breaks** | Breaks also come from switching ends, timeouts and TV edits. Las Vegas: 6 breaks > 60 s, only 1 real game end -> "4 games". The "last rally of a game = server won" rule is then applied to the wrong rallies. | Use the rules instead: rebuild the score from the serve sequence (server wins -> +1); a game ends at 11, win by 2. |
| L3 | High | **Points can be silently missing** | Nothing checks that the sequence of rallies is complete. | Singles serve rule: the server serves from the **right** when their own score is even, **left** when odd. Our feet positions show the side, so every serve checks the reconstructed score; a mismatch means a point was missed. |
| L4 | High | **Who-is-who by shirt colour only** | Fails when both players wear similar colours (Las Vegas: both white tops). | Players only change ends between games (and at 6 points in game 3). With games known (L2), the side->player mapping flips at each game; colour is only needed once. |
| L5 | Medium | **Two same-side hits: the earlier becomes a bounce** | If the opponent's hit in between was missed, a real hit is deleted. | Same net-crossing test as step 10: two crossings between them = a missed opponent hit (insert it); otherwise drop the weaker one. |
| L6 | Medium | **Volley = no bounce seen before the hit** | Bounce detection is ~83%, so a missed bounce turns a groundstroke into a "volley". | Use the 3D height at contact and the hitter's distance from the net as well. |
| L7 | Medium | **Bounce model training reads "rule bounces" from analysis.json** | Those are now the model's own output, so retraining would learn from its own predictions. | Compute rule bounces directly when training. |
| L8 | Medium | **Referee comparison matched rallies out of time order** | The comparison tool, not the analyzer: the Las Vegas numbers (65%/70%) are unreliable. | Order-preserving sequence alignment. |
| L9 | Medium | **Court detection needs a hand calibration per new court** | The court model was trained on other courts. | Short term: one-click calibration screen (4 corners). Longer: train on more courts. |
| L10 | Medium | **Hit/bounce models trained on one match** | 70 hits / 33 bounces from one venue. | Tag 5 rallies per new venue; retrain on all. |
| L11 | Low | **Fixed-camera assumption** in step 2 | A broadcast that pans/zooms would lose real frames. | Only apply the filter when the camera is clearly fixed. |
| L12 | Low | **Speed ignores height and uses feet, not the ball** | Shot speed is approximate. | Use the 3D arc for long shots; label as "approx". |
| L13 | Low | **Players named by "P1 = far side at the start"** | Names can be silently swapped if the far player at the start is guessed wrong. | Show one frame at setup: "who is this?". |
| L14 | **Critical** | **The far player is often not detected on a new broadcast** | Las Vegas: far player missing in 77% of court frames (US Open: 2%). Small (~80-130 px) player in white against a busy background; the pose model at full-frame size misses her. Every far-side hit, serve and position is lost, and whole rallies drop out. Main reason Las Vegas looked bad. | Second look in a 2x enlarged crop of the far baseline area when nobody is found beyond the net (`detect.add_far_player`); recovers ~80%. Repair tool for old runs: `training/repair_far_player.py`. |

## 3. Status of the fixes (2026-10-02 evening)

| # | Status |
|---|---|
| L1 | Done: 1-hit rallies kept when the hit looks like a serve (`segment_rallies.serve_only`). |
| L2, L3, L4 | Done (`outcome.py`, `appearance.py`): scores per player slot; END CHANGES OBSERVED from the near player's upper+lower body colours (skirt/shorts differ even when shirts match), rules-only search as fallback; game ends at 11 (win by 2) or at a long break near 11 when the first server alternates (not a timeout); best of 3, post-match play excluded; serve-position rule only as a flag (adding "missed points" from it inflated scores). Formats differ: Las Vegas switched at 6 in each game and NOT between games; US Open broadcast edited out the game break. |
| L8 | Done: order-preserving alignment in `training/compare_referee.py`. |
| L14 | Done in detection + repaired Las Vegas (far player added in 12,848/16,379 frames). |

**Results after the fixes** (server / point winner from tracking + rules only):

| Match | Rallies | Server | Winner | Games |
|---|---|---|---|---|
| US Open (Boyer-Devilliers), vs scoreboard | 62 (4 post-match excluded) | 19/19 vs tags | 56/56 | 2: 11-4, 11-3 |
| Las Vegas (Fahey-Yoshitomi), vs referee log | 48 of 53 points seen (5 not shown / missed), 1 split | 47/48 (98%) | 45/48 (94%) | 2: 11-4 (true 11-3), 9-0 (true 11-1) |
| L5, L6, L7, L9-L13 | Open. |

**Tried and rejected:** TrackNet-style 3-frame ball gap filler (own implementation, labels from YOLO + trusted
3D arcs, trained on Kaggle). Used only where YOLO has no ball, it lowered trusted 3D arcs (-4 to -5 points) and
hits found (82 -> 77 of 87) on the match it was not trained on. Not used.

## 4. What is verified vs assumed

- Verified with tags / referee / scoreboard: steps 6, 7, 9, 10-12 (US Open), court and camera (both matches).
- Partly verified: 4, 5, 14, 16.
- Not verified yet: 15 (speed), 17 (movement numbers).
