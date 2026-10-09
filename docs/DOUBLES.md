# Doubles: individual and partnership analytics

Code: `pickleball/doubles_players.py`, `training/build_doubles_players.py` -> `outputs/reports/<report>.doubles.json`
(the report itself is not changed). Shown in the match workspace (Partnership section, per-player Movement, the
individual "Analyse" views), the coaching brief and player profiles. Tests: `tests/test_doubles_players.py`.

## What other doubles sports show (2026-10-09 scan)
- Tennis doubles: formations (I-formation used on ~41-46% of points in one study), poaching by the net player,
  net position, serve / return per player; returns go cross-court 75-90% of the time
  ([tennis doubles formation study](https://ojs.ub.uni-konstanz.de/cpa/article/view/6659),
  [poaching](https://www.tennisconsult.com/poaching-like-real-doubles-player/)).
- Padel: distance per player split by rally role (server, returner and their partners cover different distances),
  left- vs right-side player roles, cross-court share
  ([Aspetar: performance in padel](https://journal.aspetar.com/en/journals/volume-12-targeted-topic-sports-medicine-in-padel/performance-in-padel),
  [distance by level](https://portalinvestigacion.um.es/documentos/5f8646f72999527b5bcb2a82)).
- Pickleball apps (PB Vision): per-player shot charts and component ratings (kitchen game, court IQ, targeting)
  ([PB Vision via Sporfie](https://en.sporfie.com/integrations/pbvision)).
We take the measurable parts: per player (shot share, side, net time, movement, rally-role arrival) and per pair
(formation, spacing, side switches, middle balls, cross-overs), always with the denominator. No ratings.

## Definitions
- Positions are seen from the player: u = 0 at their left sideline, d = feet from the net. "Up" = within 10.5 ft of
  the net (same line as singles kitchen time and net arrival; see "Up line" below); "back" = deeper than 15 ft.
- Formation per 0.1 s (both partners named): both up / one up / both back; also at every opponent contact.
- Side switch: the left-side partner changes and stays changed for 0.5 s (partners >= 2 ft apart sideways).
- Middle ball: a team rally shot (3rd shot on) hit from the centre third. Cross-over: at the opponents' previous
  contact the hitter was on one side of the partner, at this contact on the other (poach / switch).
- Rally roles: server, server's partner, returner, returner's partner (from the credited serve and return).

## Who hit the shot (doubles)
The report's hitter (named tracklet nearest the shot's hitter position) picked the kitchen-line partner too often:
right on 21 of 39 serves named by the TNPPL overlay and 7 of 45 returns decided by the depth rule (the returner is
the deeper receiver). Replaced, per shot, by: overlay server name (serves), deeper receiver when the partners are
>= 5 ft apart in depth (returns), else the partner whose image box (widened for paddle reach) is nearest the detected
ball around contact - only with a 25 px margin; otherwise the shot stays team-level ("unattributed", ~15-20%).
Check against the 3D ball contact point (where it is decisive): ball-based picks agree 152/163 (93%), overlay /
depth-rule picks 67-70%, the old method 129/225 (57%). No hand-tagged doubles hitters yet - the next validation step.

## Far-end depth (TNPPL)
The low indoor camera reads far-end players too deep. One correction is used for every position number (singles and
doubles: movement, kitchen time, net arrival, time pressure, hitter zones, placement origins from the feet, doubles
positions): pickleball/depth.py treats the foot point as sitting h ft above the floor and moves it along the camera
ray. h is calibrated per match by a game rule, first that applies:
1. Server stance (servers stand equally far behind the baseline at both ends) - when it is plausible.
2. Kitchen line (doubles): at the serve the receiving pair is one up, one back; the shallower partner (the returner's
   partner) waits at the kitchen line at both ends. Pooled over the tie's doubles matches with named players (same
   court and camera rig), solved for equal medians at both ends.
3. Default h = 0.3 ft.
training/build_tnppl.py recalibrates every match of the tie first and writes outputs/<run>/depth.json, which every
report builder reads (depth.for_run / depth.apply; analysis.json is not changed).

TNPPL KT v RTT (2026-10-09): 16 + 64 kitchen-line stances, raw 6.8 ft from the net at the near end vs 12.0 ft at the
far end -> h = 0.53 ft, both 8.9 ft. Independent check on the other end of the court: the deeper player of the
serving pair (the server) stands 2.2 ft behind the baseline at the near end vs 7.6 ft at the far end raw; with
h = 0.53 3.2 vs 3.3 ft (with the old default 0.3: 2.7 vs 5.2). Used by MS, WD, XD, MD50 and Team doubles (their
server rule fails); MD (0.61), MD 2 (0.48) and Team singles (0.23) keep their own server-rule h.
The earlier doubles-only offset (4.74 ft, far end only, on top of the default) is gone: it took the partner of the
credited returner, which mixed in deep players (returners, wrongly credited returns) and put far-end kitchen-line
players at 5.5 ft, inside the kitchen.
Effect on the doubles numbers: players now read ~1.5-2.5 ft deeper than in the first doubles build (the far end is
no longer over-corrected); "up" time and "both up" fall (e.g. Rockfort in Team doubles both up 36% -> 10%).
Later the same day TNPPL Court 1 moved to ONE value for the whole rig (h 0.53 for every TNPPL match; PROGRESS.md).

## Up line (10.5 ft, 2026-10-09)
One line for "up" (doubles), kitchen time, back-to-the-kitchen time, net arrival and the net / mid / back contact
zones: pickleball/court.py UP_FT. It was 9 ft, set on the old positions; after the depth fix the players who are
clearly at the kitchen line measured around 9 ft, so 9 ft cut them in half:
- kitchen-line waiters at the serve (TNPPL, 144): median 9.2 ft; within 9 ft 45%, 10.5 ft 76%, 11 ft 82%.
- the hitter's feet at dinks (411 dinks): median 9.3 ft; within 9 ft 45%, 10.5 ft 68%.
- all player positions in rallies pile up from the kitchen line (7 ft, sharp edge: nobody stands inside) to ~11 ft,
  peak 8-10 ft, then a flat mid-court level - the same shape at TNPPL (doubles and singles), US Open and Las Vegas.
Choice: the line that misclassifies the least rally time (kitchen cluster missed + mid-court time counted as up):
lowest at 10.5-11.5 ft at every venue, 9 ft the worst. 10.5 ft = the lower end, so the least mid-court time is
counted as up.
Effect (rebuilt 2026-10-09): kitchen time +2 to +23 points per side, net reached up to +53 points (TNPPL SSS MD50
18 -> 71%), doubles "up" +6 to +27 points, "both up" +7 to +20 points; US Open Boyer kitchen time 8 -> 16%,
Las Vegas Fahey 2 -> 8%. Scoring and placement unchanged.
Serving-team cue (pickleball/doubles.py formation) stays on raw positions: with corrected positions it agrees with
the score overlay less often where it decides (216 of 253 vs 222 of 248 raw; 12 TNPPL matches).

## Limits
- Positions about +-1 ft (the depth correction sets both ends equal; the absolute level carries ~1 ft); frames with an unnamed player are left out (named 81-97% of player frames).
- Team event (rotating players): pair numbers combine every pair the team used.
- Las Vegas doubles have no named players yet (mark once in dashboard/identify.html, then rebuild).
- Rally winners are not used (unreliable in doubles).
