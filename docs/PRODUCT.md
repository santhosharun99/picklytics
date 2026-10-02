# Product direction

**A pickleball analysis tool for coaches and analysts**: player performance and opposition
scouting from match video. Think Hudl/Sportscode-style analysis (as used in cricket and
football), not a consumer "rate my game" app. Every number must be backed by clickable video.

## Questions the tool answers

| Coach question | Output | Depends on |
|---|---|---|
| How do they serve / return? | depth & direction maps by side (deuce/ad), consistency | bounce locations |
| What do they do on the 3rd shot? | drop / drive / lob mix and success rate | shot type, point outcome |
| Where do they hit from and to? | hitting-zone heatmap, target map per shot type, FH/BH split | positions, bounce locations |
| How do they move? | coverage heatmap, time to kitchen after serve/return, depth, recovery | player tracking |
| Strengths / weaknesses | win% and error% by shot type, stroke side, zone, rally length, pressure situations | point outcomes |
| Patterns & tendencies | frequent shot sequences ("after X they do Y 40% of the time") | reliable shot sequence |
| How to beat them | game-plan summary: targets, situations to create, what to avoid | all above + AI summary |
| Show me | every stat links to the clips behind it | rally/shot timestamps |
| Scout across matches | aggregate several matches of one opponent | multi-match store |
| Fix the AI | analyst corrects tags before reporting | labelling tool (exists) |

## Roadmap

1. **Tracking accuracy** - ball model v2, hit detection toward ~90%, bounce/landing spots.
2. **Point outcomes** - rally winner and ending (winner / unforced error / forced error / fault).
3. **Player profile & scouting report** - heatmaps, serve/return & 3rd-shot profiles, target maps,
   FH/BH, movement, strengths/weaknesses, clips. Prototype on hand-labelled rallies first.
4. **Patterns** - sequence mining of shot types / directions / zones.
5. **Multi-match scouting, PDF export, AI written summary** (an LLM over the computed stats).
6. Later: doubles, phone-video input, 3D ball height for shot quality.
7. **Indian tournament database** (after the analysis tool is solid): results, draws, scores and stats of
   tournaments held in India -> player profiles, match history, head-to-head, form; feeds opponent
   shortlists for scouting. Sources to evaluate: national associations' results pages, DUPR (API, needs
   approval), tournament platforms, organisers' posts. Scope: scores and stats only, no video. Prefer official APIs or permission over scraping; respect site terms and
   robots.txt; store only public results data about players.

## Principles

- Accuracy before features: a wrong stat in a scouting report is worse than a missing one.
- Measure everything against analyst-labelled rallies (`training/evaluate.py`).
- Analyst-in-the-loop: AI drafts, the analyst reviews, the report uses reviewed data.

## Research notes (reviewed 2026-10-02)
- **ChatMatch** (Zhang et al., Computer Speech & Language 2025, open access): badminton videos -> vision models extract
  per-frame meta-features (locations, actions, gestures, results) -> rules turn them into structured match data ->
  LLM agents (task identifier, coach, statistician, video manager) answer coach questions in plain language and pull
  matching clips. Fits us later: our analysis.json + clips are the structured data; add an "ask the match" box on
  match.html ("where does Boyer lose points on the backhand?"). Does not improve detection.
- **Lightweight 3D pose estimation and matching** (Zhang et al., Neurocomputing 2026, pickleball): BlazePose keypoints
  + genetic-algorithm alignment over time to compare a player's stroke with a reference stroke, giving a match score
  and visual feedback. Fits a later "technique" module (compare a player's dink/drive form to a model stroke).
  Needs a closer camera than a broadcast; does not improve hit detection.
