# Research roadmap (built one research track at a time; sources checked via Crossref)

## Track 01 - Performance indicators (Gemini result 01, reviewed 2026-10-03)
Sources verified (DOI exists, title matches): Escudero-Tena et al. 2022 Applied Sciences (padel errors/winners);
Torres-Luque et al. 2015 IJPAS (padel match analysis); Prieto-Lage et al. 2024 Applied Sciences (men's singles
pickleball notational analysis - Gemini mis-attributed it to "Wang et al."); Lozano et al. 2025 IJPAS (pro doubles
pickleball activity profile); Filipcic et al. 2015 JPES (DOI exists, title not in Crossref).
- Confirms metrics we already show: won on serve / on return, win % by rally length (pickleball singles evidence), speed,
  distance covered.
- BUILD: kitchen-line (net position) control - share of time / shots near the kitchen line (padel evidence); needs the
  far-end depth correction first.
- BUILD: third-shot effectiveness by pace band, landing depth and net clearance (3D lifter) vs rally won.
- DATA NEEDED: rally-ending tags (winner / forced / unforced error, net / out) -> unforced error rate and
  winner:error ratio (strong padel/tennis evidence). Tagging task for the labeller in label.html.
- SKIP: "first serve %" (tennis-only), work:rest ratio (broadcast edits distort rest time).

## Track 02 - Tactics, rally models, win probability (Gemini result 02, reviewed 2026-10-03)
Sources: Walker & Wooders 2001 AER "Minimax Play at Wimbledon" (verified); momentum paper verified but authors are
Ding et al. 2026 (Gemini: "Shen 2024"); T-pattern paper verified, author Magnusson (Gemini: "Amatria"); "Rhythms of
Victory" IEEE Access 2024 verified, Lei et al. (Gemini: "Li"); arXiv 2005.12853 (tennis decision-making, VON CRAMM)
title verified; Newton & Keller 2005 and Wei et al. 2016 TKDE are known papers but Gemini's DOIs do not resolve.
Lesson: Gemini's author names are often wrong -> always verify.
- BUILD (Tier 1): side-out-scoring Markov model -> win-probability graph per game + point importance (leverage);
  performance on high-importance points. Inputs: server, winner, score (94-100%). Use pooled/shrunk serve-point
  probabilities (see track 09) because one match is small.
- BUILD when several matches per player (Tier 2): serve-direction predictability / minimax test (lateral serve
  landing is a reliable axis); shot-direction entropy per situation (pooled).
- BUILD after depth fix (Tier 2): court control as "how far the opponent had to move to reach each ball".
- LATER (Tier 3, data volume): rally shot-value model (EPV-style, see track 06), T-pattern mining (needs better shot
  labels), deep next-shot forecasting (track 10).
- SKIP for now: momentum score (contested evidence, misleading risk).

## Tracks 03-11 (Gemini did all 9 in one go, reviewed 2026-10-04)
Source quality dropped: only 14 DOIs for 72 ideas; 11 verified (e.g. Mlakar & Kovalchik 2020 J Sports Analytics,
time pressure in tennis; Giles et al. 2024 JSCR, change of direction in pro tennis; Decroos et al. 2019 KDD, VAEP;
Prieto-Lage et al. 2024 pickleball singles; Lozano et al. 2025 pickleball doubles; several pickleball injury studies);
3 do not resolve; track 10 cites an unrelated graph-learning paper. Many ideas cite only "standard practice" -> treated as
unverified. Well-known real methods referenced without DOI (checked by name): Spearman 2018 pitch control (SSAC), Le et al.
2017 data-driven ghosting (SSAC), Fernandez & Bornn 2018 (SSAC), Cervone et al. 2014 EPV (SSAC), Franks et al. 2015
(SSAC), Glickman 2001 Glicko-2, Herbrich et al. 2006 TrueSkill (NeurIPS), Wang et al. 2022 ShuttleNet (AAAI), Decroos &
Davis 2019 Player2Vec, Singh 2018 expected threat.

# CONSOLIDATED ROADMAP (all 11 tracks)

## Foundations (do first)
1. Far-end depth correction from game rules (server stance, serves land in the diagonal box).
2. Honest small-sample statistics everywhere: empirical-Bayes shrinkage + ranges for every rate (track 09).
3. Rally-ending tags (winner / forced / unforced, net / out) in label.html - unlocks errors/winners (tracks 01, 04, 08).
4. MORE MATCHES: the single biggest need for tiers 2-3 (10-20+ broadcast matches across venues; a few tagged rallies
   per new venue). Pairs with the planned Indian-tournament scores database for ratings.

## Tier 1 - strong basis, our data supports it (now or after foundation 1)
- Time pressure (Mlakar & Kovalchik 2020): time between the opponent's hit and the player's hit + distance moved;
  who forces whom to play rushed shots, and win % under high pressure.
- Win probability + point importance (track 02, Markov for side-out scoring): big-point performance.
- Third-shot effectiveness by pace / depth / net clearance vs rally won (Prieto-Lage 2024).
- Kitchen-line (net) control and first-to-the-net (padel evidence, track 01) - after depth fix.
- Change-of-direction load (Giles et al. 2024) from 10 Hz court tracks - check noise first.

## Tier 2 - needs several matches per player or a foundation
- Serve predictability / minimax test (Walker & Wooders 2001) and shot-direction entropy per situation.
- Court control / time-to-reach (pitch-control idea, Spearman 2018, adapted to 2 players): how far each ball pulls the
  opponent and whether they got there in time.
- Player style fingerprints: Markov transition profiles first, embeddings (Player2Vec) once many matches exist;
  similarity search "plays like X".
- Player ratings with uncertainty (Glicko-2 / TrueSkill / component serve-return ratings) across matches and the
  scores database.

## Tier 3 - needs much more data or labels
- Shot value models (VAEP / EPV / expected threat: Decroos 2019, Cervone 2014, Singh 2018): "how much did this shot
  change the chance of winning the point"; needs thousands of rallies.
- Decision quality (process vs outcome, counterfactual alternatives) - built on top of a shot value model.
- Ghosting / ideal positioning (Le et al. 2017), deep next-shot prediction (ShuttleNet, Wang et al. 2022).

## Not pursuing (not measurable from one broadcast camera, or weak evidence)
Injury / orthopedic stress scores and paddle-speed load (medical claims, not measurable), momentum scores, Markowitz
"risk frontier", left-handedness effects (sample), reinforcement-learning valuation (data volume), work:rest ratio
(broadcast edits distort rest time).

## Tracks 03-11, second pass (Gemini "deep research" rewrite, checked 2026-10-04)
41 DOIs (40 unique), each checked against Crossref (title + first author) or arXiv:
- 29 correct and on topic (some with wrong year/author, e.g. Prieto-Lage 2024 again cited as "Wang et al.").
- 4 resolve to an UNRELATED paper: 10.1371/journal.pone.0310706 (diabetes costs, Nepal; cited as tennis
  technical-tactical diversity), 10.3390/s23156686 (nitrous-oxide sensor; cited as soccer playing-styles review),
  10.1111/1467-9876.00155 (cited as Glickman 1999; real Glickman paper is a different DOI), 10.1214/23-AOAS1768
  (cited as NBA timeout causal study). 10.3390/ijerph17217838 is a real padel paper but not the title cited.
- 6 do not resolve (two repeated from the first pass, incl. the "Wei 2016 TKDE" tennis paper).
- Several real ones are from weak venues (Advances in Nano Research, Academic Journal of Computing & Information
  Science, IJIRMPS); treated as ideas, not evidence.
Conclusion: the "every indicator cites a precise DOI" claim is not true; about 3 in 10 citations fail. Keep verifying.

New ideas with a real, on-topic source that change the plan:
- Movement efficiency / competition load (distance per minute, distance per shot): Miralles et al. 2026 Biol Sport
  (elite women's padel) -> Tier 1, our court tracks support it now.
- Shot risk-reward frontier: Fichman & O'Brien 2019 JQAS (NBA optimal shot selection) -> moved from "not pursuing"
  to Tier 2 (needs rally-ending tags + several matches).
- Mechanical work as a fatigue indicator: Emmerson et al. PLoS ONE (tennis, markerless capture) -> Tier 2; needs
  per-match movement noise check first.
- Reachability / time-to-intercept per shot: fits inside the Tier 2 court-control item (Martens 2021 Front Sports, soccer
  space control, is the verified source).
- Shot diversity score (technical-tactical diversity): source DOI is wrong, but the idea is cheap (entropy of shot
  types/directions) -> fold into the Tier 2 predictability/entropy item.
Unchanged: VAEP/EPV/xT (verified: Decroos 2019, Cervone 2016 JASA, Fernandez 2021 Mach Learn, Hassani 2025 DxT) and
ShuttleNet stay Tier 3; injury/load, momentum and EWM-TOPSIS scores stay out.

## Built so far (2026-10-04)
Foundation 1 (depth fix), replay filter, time pressure, movement efficiency, win probability + importance + big points,
third-shot quality. Findings on our 2 matches: rushed shots -> more errors (22% vs 15%); third-shot drive -> server
wins 61% vs 19% (p<0.001) [RETRACTED 2026-10-04: artifact of rule-based shot types]; per-player big-point / pressure differences not significant yet (p~0.3) -> shown as
numbers only. Remaining Tier 1: kitchen-line control (first to the net), change-of-direction load.
