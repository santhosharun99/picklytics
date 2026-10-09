# Placement & speed (primary shot analysis)

Measured shot characteristics come first; automatic shot-type labels (drive, drop, reset ...) are secondary and
qualified. Code: `pickleball/zones.py` (geometry, the only place for bands/orientation), `pickleball/shot_events.py`
(per-shot event record), `pickleball/report.py` (per-shot `pl` field in every report shot), `pickleball/player_db.py`
(facts per player, singles and doubles apart), `dashboard/match.html` (Placement & speed section),
`dashboard/player.html`. Tests: `tests/test_zones.py`, `tests/test_shot_events.py`. Validation:
`training/eval_placement.py`.

## Zones and orientation
- Court coordinates: x 0..20 across (0 = left sideline seen from the near baseline), y 0 (far baseline) .. 44 (near
  baseline), net y = 22.
- Every location of a shot is described **from the hitter, looking over the net**: lateral u = x for a near-end
  hitter, u = 20 - x for a far-end hitter (0 = the hitter's left). The end comes from the hitter's tracked side for
  that shot, so end changes between games are handled shot by shot. Depth d = |y - 22| (ft from the net).
- Bands, half-open (lower bound included): kitchen [0, 7), midcourt [7, 14.5), deep [14.5, 22]; left [0, 20/3),
  centre [20/3, 40/3), right [40/3, 20]. Lines are in (d = 22 and u = 20 are inside). Names: "kitchen left" ...
  "deep right".
- Outside: d > 22 -> "behind ...", u < 0 / u > 20 -> "wide-left" / "wide-right". Never clamped into an inside zone.
  An origin behind the baseline is a normal origin; an estimated landing outside is an estimate, not a line call.
- Uncertainty: with a 1-sigma position sd, every zone within +-1 sd is a candidate; more than one candidate -> zone
  "boundary" (candidates kept). Aggregates count only zone-certain locations and show how many were boundary. The
  sd comes from the physics fit covariance; it is a model quantity, not a calibrated error bar.

## Events of a shot
hitter contact -> outgoing flight -> first event, inside the window up to the next contact (the next shot). Nothing
observed after the next contact belongs to the shot. Post-point ball handling is excluded by the rally-end model
(dead touches are not shots).
- **bounced**: an observed bounce on the opponent's half comes first. It is the observed landing (a later opponent
  contact is the next shot).
- **intercepted**: the opponent hit it before any bounce - only when the next contact is flagged a volley AND the
  ball was tracked densely through the flight (>= 60% of frames, no gap > 0.2 s) so a bounce could have been seen,
  AND the rules allow it (serve and return must bounce).
- **net**: a physics 'net' junction comes first and no bounce follows on the opponent's half; no extrapolation through the net. A net touch followed by an opponent-half bounce is a **net cord**: bounced (flag `net_cord`), the bounce is the landing. Taggers mark only the bounce after a net cord.
- **unresolved**: everything else (missed bounce, sparse tracking, a bounce on the hitter's own half without a net
  event, rally ended with nothing observed). Never forced into net / in / out. The weak ending classifier is not used.

Bounce sources, in order: human tags (inside hand-tagged rallies they are the complete record), the physics fit
(pickleball/rally3d.py bounce junctions), the bounce detector. Physics + detector within 4 frames = one bounce,
quality "two automatic sources agree".

## Four separate locations
| field | meaning | evidence |
|---|---|---|
| origin | where the shot was hit | ball at contact (3D fit) or the hitter's feet (player-position proxy, labelled) |
| landing | observed first bounce | state observed / not-applicable (intercepted, net) / unknown (unresolved) |
| interception | where the opponent hit it before a bounce | ball at the opponent's contact (3D fit) or the opponent's feet (proxy, labelled) |
| projected | counterfactual landing of an intercepted ball | **unknown for now**: no validated projection model |

An interception is never counted as a landing. The landing map is conditional on shots that bounced first - it is
not the placement of all shots.

## Speed
One definition everywhere: **estimated launch speed off the paddle** = initial speed of the physics fit of the
outgoing flight (drag + per-flight effective gravity, pickleball/rally3d.py), using only observations before the
next contact. Used when the flight is physically plausible and >= 6 ball positions were seen; else unknown. Shown in
whole mph with the usable count. Not checked against radar. The older speed module (`ball_mph`, stopwatch + arc)
measures the same quantity with shared inputs, so the two agreeing is not independent validation; the "+-3-4 mph"
claim was removed. Aggregates never mix the two.

## Validation (US Open, the only match with tagged bounce positions)
`PYTHONPATH=. python3 training/eval_placement.py` - automatic sources only vs the hand-tag version, 82 shots in 15
tagged rallies:
- observed landings: 35 of 36 tagged landings found; distance median 0.53 ft, 80% within 1.17 ft; zone same 27,
  boundary 5, different 3 (of 35);
- outgoing-flight status: 60 of 82 agree (73%); 15 of the 22 differences are bounces after the last tagged event of
  the rally (the tags stop there) -> 60 of 67 (90%) inside the tagged stretch.
TNPPL (indoor, low camera; Rahul vs Ramakrishnan, 15 rallies tagged 2026-10-09; end change after rally 8 fixed from
tracking + the tagger; 10 flights where the bounce was not visible are flagged `next_bounce_unknown` and skipped;
positions filled automatically carry `xy_src: auto ...` and are never used as truth):
`PYTHONPATH=. python3 training/eval_placement.py tnppl_1_MS labels/TNPPL_S2_KTvRTT_1_MS.json`
- observed landings: 32 of 36 found; distance median 1.01 ft, 80% within 2.64 ft; zone same 22, boundary 2,
  different 8 (of 32) - clearly weaker than the US Open broadcast camera;
- status: 43 of 54 agree (80%); 43 of 50 (86%) leaving out 4 bounces after the last tagged event.
These are agreement with human labels on two venues. Unit tests check the rules; the physics-vs-detector agreement
is model-to-model consistency, not accuracy. Las Vegas and Europe are not validated.

## Known limitations
- Projected landing not implemented (needs a holdout test: hide the end of flights with known bounces, predict,
  compare). Until then it stays unknown.
- Interception needs the automatic volley flag; a wrong flag leaves the shot unresolved rather than intercepted.
- Origins from the 3D fit can be off for contacts far from the camera; 1 in 5 origins is a feet proxy.
- Feet-proxy origins use the player depth correction (pickleball/depth.py). Its TNPPL recalibration (kitchen-line
  rule, 2026-10-09; docs/DOUBLES.md) moved 31 of 922 origins into another zone, all feet proxies (mostly kitchen ->
  midcourt and behind -> deep); 3D-fit origins and landings do not depend on it.
- Doubles: shots are credited to the named hitter where identity exists; team-level otherwise. Rally outcomes are
  not used in placement facts.
