"""Player database across all match reports -> dashboard/players.json (pickleball/player_db.py).

    PYTHONPATH=. python3 training/build_players.py
Re-run after rebuilding any report.
"""
import json
from pathlib import Path

from pickleball.player_db import build

ROOT = Path(__file__).resolve().parents[1]
db = build(ROOT)
(ROOT / "dashboard" / "players.json").write_text(json.dumps(db, indent=1, ensure_ascii=False))
for p in db["players"]:
    print(f"{p['name']:20s} {p['team'] or '-':20s} matches {len(p['matches'])} shots {p['stats']['shots']}")
