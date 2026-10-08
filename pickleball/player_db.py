"""Player database across matches: every named player's shots from all reports -> one profile + a plain-language
summary of how they play, compared with the other players we have (dashboard/players.json, player.html).

Who hit a shot: singles = the report's player; doubles = the named hitter (mark-once identity, ~98% right);
TNPPL team singles rotates players without names -> left out (team level only).
"""
import json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TOPSPIN_G, SLICE_G = 40.0, 25.0          # same cut-offs as report.spin_profile (experimental)
SOFT = ("dink", "drop", "reset", "third-shot drop")
DRIVE = ("drive", "third-shot drive")
ALIASES = {"Sivasankar Shivan": "Sivashankar Sivan"}  # same TNPPL player, spelled two ways in the sources
MIN_SHOTS = 15                            # below this a comparison is called a small sample


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _team_of(root: Path) -> dict:
    cfg = root / "configs" / "tnppl_s2_kt_rtt.json"
    if not cfg.exists():
        return {}
    C = json.loads(cfg.read_text())
    names = {"KT": "Kodai Tigers", "RTT": "Rockfort Terminatrz"}
    return {ALIASES.get(p, p): names.get(k, k) for k, ps in C.get("rosters", {}).items() for p in ps}


def _shots_of(R: dict, name: str, doubles: bool):
    for r in R["rallies"]:
        for s in r["shots"]:
            who = s.get("hitter") if doubles else s.get("player")
            if who and ALIASES.get(who, who) == name and not s.get("inferred"):
                yield r, s


def build(root: Path) -> dict:
    root = Path(root)
    M = json.loads((root / "dashboard" / "matches.json").read_text())
    T = {t["id"]: t for t in M.get("tournaments", [])}
    team_of = _team_of(root)
    shots, matches = defaultdict(list), defaultdict(list)
    insights = defaultdict(list)
    singles_prof = defaultdict(list)
    for m in M["matches"]:
        if not m.get("report"):
            continue
        f = root / "outputs" / "reports" / f"{m['report']}.json"
        if not f.exists():
            continue
        R = json.loads(f.read_text())
        doubles = R.get("type") == "doubles"
        if doubles:
            people = {ALIASES.get(n, n) for n in R.get("individuals", {})}
            side_of = {ALIASES.get(p, p): t for t, ps in (R.get("teams") or {}).items() for p in ps}
        elif m["id"] == "tnppl_8_TeamSingles":
            continue  # rotating unnamed players
        else:
            people = set(R["players"])
            side_of = {p: p for p in people}
        sides = list(R["players"])
        for name in people:
            mine = list(_shots_of(R, name, doubles))
            if not mine:
                continue
            side = side_of.get(name)
            opp = next((x for x in sides if x != side), None)
            played = {r["id"] for r, _ in mine} if doubles else {r["id"] for r in R["rallies"]}
            won = sum(1 for r in R["rallies"] if r["id"] in played and r.get("winner") == side)
            decided = sum(1 for r in R["rallies"] if r["id"] in played and r.get("winner"))
            partner = [p for p in (R.get("teams") or {}).get(side, []) if ALIASES.get(p, p) != name] if doubles else []
            opponents = (R.get("teams") or {}).get(opp, []) if doubles else [opp]
            matches[name].append({"id": m["id"], "title": m["title"], "round": m.get("round"), "type": m["type"],
                                  "tournament": T.get(m.get("tournament"), {}).get("name", m.get("event")),
                                  "team": side if doubles else None, "partner": partner, "opponents": opponents,
                                  "shots": len(mine), "rallies": len(played),
                                  "won_pct": round(100 * won / decided) if decided else None, "official": R.get("official")})
            for r, s in mine:
                shots[name].append({**s, "_match": m["id"], "_rally": r["id"]})  # includes s['pl'] (placement record)
            if not doubles:
                P = R["players"][name]
                singles_prof[name].append((m["id"], P))
                for i in P.get("insights", []):
                    if i["kind"] in ("strength", "weakness", "pattern"):
                        insights[name].append({**i, "match": m["id"], "match_title": m["title"]})

    def stats(ss):
        n = len(ss)
        typed = [s for s in ss if s.get("shot")]
        mix = Counter(s["shot"] for s in typed)
        stroke = Counter(s["stroke"] for s in ss if s.get("stroke"))
        zones = Counter(s["zone"] for s in ss if s.get("zone"))
        rally_shots = [s for s in typed if s["shot"] not in ("serve", "return")]
        vol = sum(1 for s in rally_shots if s.get("volley"))
        sp = lambda kinds: [s["speed_mph"] for s in ss if s.get("speed_mph") and s.get("speed_conf") in ("high", "medium")
                            and (kinds is None or s.get("shot") in kinds)]
        spin = [s["spin_g"] for s in ss if s.get("spin_g") is not None]
        spin_dr = [s["spin_g"] for s in ss if s.get("spin_g") is not None and s.get("shot") in DRIVE]
        clr_soft = [s["phys_clear_ft"] for s in ss if s.get("phys_clear_ft") is not None and s.get("shot") in SOFT and s["phys_clear_ft"] > -1]
        clr_dr = [s["phys_clear_ft"] for s in ss if s.get("phys_clear_ft") is not None and s.get("shot") in DRIVE and s["phys_clear_ft"] > -1]
        deep = [abs(s["land"][1] - 22) >= 15 for s in ss if s.get("land") and s.get("shot") in DRIVE + ("return", "serve")]
        med = lambda v: round(float(np.median(v)), 1) if v else None
        pct = lambda a, b: round(100 * a / b) if b else None
        return {
            "shots": n, "typed": len(typed),
            "mix": {k: pct(v, len(typed)) for k, v in mix.most_common()},
            "mix_n": dict(mix.most_common()),
            "forehand_pct": pct(stroke.get("forehand", 0), sum(stroke.values())), "stroke_n": sum(stroke.values()),
            "zones": {k: pct(zones.get(k, 0), sum(zones.values())) for k in ("baseline", "transition", "kitchen")}, "zones_n": sum(zones.values()),
            "volley_pct": pct(vol, len(rally_shots)), "rally_shots_n": len(rally_shots),
            "drive_pct": pct(sum(mix.get(k, 0) for k in DRIVE), len(rally_shots)),
            "soft_pct": pct(sum(mix.get(k, 0) for k in SOFT), len(rally_shots)),
            "lob_pct": pct(mix.get("lob", 0), len(rally_shots)),
            "speed": {"serve": med(sp(("serve",))), "return": med(sp(("return",))), "drive": med(sp(DRIVE)),
                      "soft": med(sp(SOFT)), "all": med(sp(None))},
            "speed_n": {"serve": len(sp(("serve",))), "return": len(sp(("return",))), "drive": len(sp(DRIVE)),
                        "soft": len(sp(SOFT)), "all": len(sp(None))},
            "spin": {"n": len(spin), "topspin": pct(sum(g >= TOPSPIN_G for g in spin), len(spin)),
                     "slice": pct(sum(g <= SLICE_G for g in spin), len(spin)),
                     "drive_topspin": pct(sum(g >= TOPSPIN_G for g in spin_dr), len(spin_dr)), "drive_n": len(spin_dr)},
            "clear": {"soft": med(clr_soft), "soft_n": len(clr_soft), "drive": med(clr_dr), "drive_n": len(clr_dr)},
            "deep_pct": pct(sum(deep), len(deep)), "deep_n": len(deep),
        }

    players = []
    for name, ss in shots.items():
        types = {m["id"]: m["type"] for m in matches[name]}
        players.append({"id": slug(name), "name": name, "team": team_of.get(name),
                        "tournaments": sorted({x["tournament"] for x in matches[name]}),
                        "matches": matches[name], "stats": stats(ss),
                        "placement": {t: placement(name, [x for x in ss if types.get(x["_match"]) == t]) for t in ("singles", "doubles")
                                      if any(types.get(x["_match"]) == t for x in ss)},
                        "singles": [_singles_extra(mid, P) for mid, P in singles_prof.get(name, [])],
                        "insights": insights.get(name, [])})
    _describe(players)
    players.sort(key=lambda p: (-p["stats"]["shots"], p["name"]))
    return {"players": players, "n": len(players),
            "note": "Doubles shots are credited to the named hitter (identity ~98% right). Shot types, spin and net "
                    "clearance carry the same accuracy notes as the match reports; spin is experimental."}


ROLE_NAME = {"serve": "serve", "return": "return", "third": "third-shot", "later": "later-shot"}


def _zone_words(z: str, landing: bool = False) -> str:
    """Zone name for a sentence; outside zones say so (an estimate, not a line call)."""
    depth, width = z.split(" ", 1)
    if depth == "behind":
        return f"past the baseline ({width}) - estimated out" if landing else f"behind the baseline ({width})"
    if width.startswith("wide"):
        return f"wide of the {width.split('-')[1]} sideline ({depth}) - estimated out" if landing else f"wide of the court ({depth})"
    return z


def placement(name: str, ss: list) -> dict:
    """Placement & speed facts from the per-shot records (pickleball/shot_events.py), with denominators; one match
    type at a time (singles and doubles are never pooled). Clips: (match, rally, t) per fact."""
    pl = [s for s in ss if s.get("pl") and not s.get("inferred")]
    st = Counter(s["pl"]["status"] for s in pl)
    land = [s for s in pl if s["pl"]["status"] == "bounced" and (s["pl"].get("landing") or {}).get("state") == "observed"]
    landz = [s for s in land if s["pl"]["landing"]["zone"] != "boundary"]
    inter = [s for s in pl if s["pl"]["status"] == "intercepted"]
    icz = [s for s in inter if (s["pl"].get("interception") or {}).get("state") == "observed" and s["pl"]["interception"]["zone"] != "boundary"]
    org = [s for s in pl if s["pl"].get("origin") and s["pl"]["origin"]["zone"] != "boundary"]
    sv = [s for s in pl if (s["pl"].get("speed") or {}).get("mph") is not None]
    clip = lambda xs: [[x["_match"], x["_rally"], x.get("t")] for x in xs[:6]]
    facts = []
    def fact(text, n, xs):
        facts.append({"text": text, "n": n, "small": n < MIN_SHOTS, "kind": "fact", "clips": clip(xs)})
    if pl:
        fact(f"Of {len(pl)} analysed shots: {st.get('bounced', 0)} bounced before the opponent's contact, {st.get('intercepted', 0)} were "
             f"intercepted, {st.get('net', 0)} ended at the net, {st.get('unresolved', 0)} are unresolved.", len(pl), [])
    for role in ("serve", "return", "third", "later"):
        rz = [s for s in landz if s["pl"]["role"] == role]
        if len(rz) >= 5:
            z, c = Counter(s["pl"]["landing"]["zone"] for s in rz).most_common(1)[0]
            if c >= 2:
                fact(f"Of {len(rz)} usable {ROLE_NAME[role]} bounces, {c} landed {_zone_words(z, True)}.", len(rz), [s for s in rz if s["pl"]["landing"]["zone"] == z])
    kz = [s for s in icz if s["pl"]["interception"]["zone"].startswith("kitchen")]
    if inter:
        fact(f"{len(inter)} shots were intercepted by the opponent" + (f"; {len(kz)} of the {len(icz)} with a certain zone near the kitchen." if len(icz) >= 3 else "."),
             len(inter), kz or inter)
    if len(sv) >= 5:
        sv_i = [s for s in sv if s["pl"]["status"] == "intercepted"]
        fact(f"Median estimated speed on {len(sv)} measurable shots: {np.median([s['pl']['speed']['mph'] for s in sv]):.0f} mph"
             + (f" (on {len(sv_i)} intercepted shots: {np.median([s['pl']['speed']['mph'] for s in sv_i]):.0f} mph)." if len(sv_i) >= 5 else "."),
             len(sv), sorted(sv, key=lambda s: -s["pl"]["speed"]["mph"]))
    if len(org) >= 5:
        bh = [s for s in org if s["pl"]["origin"]["zone"].startswith("behind")]
        z, c = Counter(s["pl"]["origin"]["zone"] for s in org).most_common(1)[0]
        fact(f"Hits from behind the baseline on {len(bh)} of {len(org)} shots with a certain origin; most often from {_zone_words(z)} ({c}).", len(org), [s for s in org if s["pl"]["origin"]["zone"] == z])
    return {"n": len(pl), "status": dict(st), "landing_usable": len(landz), "landing_observed": len(land),
            "landing_zones": dict(Counter(s["pl"]["landing"]["zone"] for s in landz)),
            "interception_zones": dict(Counter(s["pl"]["interception"]["zone"] for s in icz)), "intercepted": len(inter),
            "speed_n": len(sv), "speed_median": round(float(np.median([s["pl"]["speed"]["mph"] for s in sv]))) if sv else None,
            "facts": facts}


def _singles_extra(mid: str, P: dict) -> dict:
    """Singles-only facts for one match (movement, net, time pressure), straight from the match report."""
    rn, nt = P.get("running") or {}, P.get("net") or {}
    pf, pa = (P.get("pressure") or {}).get("faced") or {}, (P.get("pressure") or {}).get("applied") or {}
    sm = P.get("summary") or {}
    v = lambda k: (sm.get(k) or {}).get("value")
    return {"match": mid, "won_pct": v("rallies_won_pct"), "won_on_serve_pct": v("won_on_serve_pct"),
            "won_on_return_pct": v("won_on_return_pct"),
            "dist_per_rally_ft": rn.get("dist_per_rally_ft"), "top_sprint_mph": rn.get("top_sprint_mph"),
            "net_reached_pct": nt.get("reached_pct"), "first_to_net_pct": nt.get("first_pct"),
            "rushed_pct": pf.get("rushed_pct"), "error_when_rushed_pct": pf.get("error_when_rushed_pct"),
            "error_otherwise_pct": pf.get("error_otherwise_pct"), "rushes_opp_pct": pa.get("rushed_opponent_pct")}


def _rank(players, key, p):
    """Where p stands among players with enough data: 'top'/'bottom' third, else None."""
    vals = [(q["id"], key(q)) for q in players if key(q) is not None]
    if len(vals) < 6 or key(p) is None:
        return None
    v = sorted(x for _, x in vals)
    r = sum(x < key(p) for x in v) / (len(v) - 1)
    return "top" if r >= 2 / 3 else ("bottom" if r <= 1 / 3 else None)


def _describe(players):
    """Plain-language 'how they play' lines from each player's stats, relative to everyone else. Each line keeps its
    sample size; lines with fewer than MIN_SHOTS shots behind them are marked small."""
    big = [p for p in players if p["stats"]["shots"] >= MIN_SHOTS]
    G = len(big)  # comparison group: players with enough shots (pros and TNPPL together - different levels)
    for p in players:
        S, out = p["stats"], []
        def add(text, n, kind="label"):  # from automatic shot-type labels / older measures: secondary
            out.append({"text": text, "n": n, "small": n < MIN_SHOTS, "kind": kind})
        z = S["zones"]
        if S["zones_n"]:
            if (z["kitchen"] or 0) + (z["transition"] or 0) >= 55:
                add(f"Plays forward: {z['transition'] + z['kitchen']}% of contacts are inside the baseline area (transition or kitchen).", S["zones_n"])
            elif (z["baseline"] or 0) >= 75:
                add(f"Mostly a back-court player: {z['baseline']}% of contacts from the baseline area.", S["zones_n"])
            else:
                add(f"Mixes court positions: {z['baseline']}% of contacts from the back, {z['transition']}% from mid-court, {z['kitchen']}% at the kitchen.", S["zones_n"])
        if S["rally_shots_n"]:
            r_drive = _rank(big, lambda q: q["stats"]["drive_pct"], p)
            r_soft = _rank(big, lambda q: q["stats"]["soft_pct"], p)
            if r_drive == "top" and (S["drive_pct"] or 0) >= (S["soft_pct"] or 0):
                add(f"Power over touch: drives are {S['drive_pct']}% of rally shots (after the serve and return) - among the most of the {G} players here.", S["rally_shots_n"])
            elif r_soft == "top":
                add(f"Soft-game player: dinks, drops and resets are {S['soft_pct']}% of rally shots - among the most of the {G} players here.", S["rally_shots_n"])
            else:
                add(f"Balanced shot choice: drives {S['drive_pct']}%, dinks / drops / resets {S['soft_pct']}% of rally shots.", S["rally_shots_n"])
            if (S["lob_pct"] or 0) >= 8:
                add(f"Uses the lob: {S['lob_pct']}% of rally shots.", S["rally_shots_n"])
            if S["volley_pct"] is not None:
                rv = _rank(big, lambda q: q["stats"]["volley_pct"], p)
                if rv == "top":
                    add(f"Takes the ball early: volleys {S['volley_pct']}% of rally shots (out of the air) - one of the highest of the {G} players here.", S["rally_shots_n"])
                elif rv == "bottom":
                    add(f"Lets the ball bounce: only {S['volley_pct']}% of rally shots are volleys.", S["rally_shots_n"])
        if S["forehand_pct"] is not None and S["stroke_n"]:
            fh = S["forehand_pct"]
            if fh >= 58:
                add(f"Leans on the forehand: {fh}% of strokes are forehands.", S["stroke_n"])
            elif fh <= 42:
                add(f"Leans on the backhand: {100 - fh}% of strokes are backhands.", S["stroke_n"])
            else:
                add(f"Even forehand / backhand split ({fh}% forehand).", S["stroke_n"])
        sd, nd = S["speed"]["drive"], S["speed_n"]["drive"]
        if sd is not None and nd >= 5:
            r = _rank(big, lambda q: q["stats"]["speed"]["drive"] if q["stats"]["speed_n"]["drive"] >= 5 else None, p)
            add(f"Drives at about {sd:.0f} mph" + (f" - one of the hardest hitters of the {G} players here." if r == "top" else f" - on the softer side of the {G} players here." if r == "bottom" else "."), nd)
        ss, ns = S["speed"]["serve"], S["speed_n"]["serve"]
        if ss is not None and ns >= 5:
            add(f"Serves at about {ss:.0f} mph.", ns)
        if S["spin"]["drive_n"] >= 6 and S["spin"]["drive_topspin"] is not None:
            add(f"Topspin on {S['spin']['drive_topspin']}% of drives (experimental spin reading).", S["spin"]["drive_n"], "experimental")
        if S["clear"]["soft_n"] >= 5:
            add(f"Soft shots cross about {S['clear']['soft']} ft above the net (experimental).", S["clear"]["soft_n"], "experimental")
        if S["deep_n"] >= 8:
            add(f"Hits deep (back 15 ft) on {S['deep_pct']}% of serves, returns and drives.", S["deep_n"])
        p["style"] = out
