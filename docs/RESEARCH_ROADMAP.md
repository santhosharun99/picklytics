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
