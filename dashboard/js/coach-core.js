// Shared, page-independent logic for the coach workspace (hub.html), the coaching brief (scout.html) and player
// profiles (player.html): which shots belong to a player/team, filters, zone aggregation, estimated-speed statistics,
// clip timing, URL state and automatic observations. No DOM access here, so it runs in Node for tests
// (tests/js/coach-core.test.mjs). Definitions follow docs/PLACEMENT.md and the per-shot `pl` records written by
// pickleball/shot_events.py - this file only counts them, it never re-derives a zone or a speed.
(function (root) {
"use strict";

const ROLES = [["all", "All shots"], ["serve", "Serve"], ["return", "Return"], ["third", "Third shot"], ["later", "Later rally shots"]];
const ROLE_NOUN = { all: "shots", serve: "serves", return: "returns", third: "third shots", later: "later rally shots" };
const ROLE_SHORT = { serve: "Serve", return: "Return", third: "Third", later: "Later" };
const STATUS_LABEL = { bounced: "bounced first", intercepted: "intercepted", net: "ended at the net", unresolved: "unresolved" };

// The three map views never share a population: a landing is an observed first bounce, an interception is the
// opponent's contact before any bounce, an origin is where the shot was hit.
const VIEWS = {
  landing: { label: "Landings", key: "landing", half: "opponent", noun: "recorded landings", one: "recorded landing",
    population: s => s.pl.status === "bounced", popNoun: "shots that bounced first",
    def: "Observed first bounce on the opponent's half." },
  interception: { label: "Interceptions", key: "interception", half: "opponent", noun: "recorded interceptions", one: "recorded interception",
    population: s => s.pl.status === "intercepted", popNoun: "intercepted shots",
    def: "Where the opponent hit the ball before it bounced. Not a landing." },
  origin: { label: "Hit from", key: "origin", half: "own", noun: "recorded contact points", one: "recorded contact point",
    population: () => true, popNoun: "shots",
    def: "Where the shot was hit: the ball at contact (3D fit) or, where the ball was not fitted, the hitter's feet (position proxy)." },
};

const DEPTHS = ["kitchen", "midcourt", "deep"], WIDTHS = ["left", "centre", "right"];
// rows from the top of the drawing to the bottom (viewer behind the hitter, looking over the net)
const ROWS = { landing: ["deep", "midcourt", "kitchen"], interception: ["deep", "midcourt", "kitchen"], origin: ["kitchen", "midcourt", "deep", "behind"] };
const BANDS = { kitchen: [0, 7], midcourt: [7, 14.5], deep: [14.5, 22], behind: [22, 29] };
const cellZones = view => ROWS[view].flatMap(d => WIDTHS.map(w => `${d} ${w}`));
const isCell = (zone, view) => cellZones(view).includes(zone);

// Evidence filter options differ per view (they describe that view's location).
const EVIDENCE = {
  landing: [["human", "Human-reviewed"], ["auto2", "Automatic · two sources agree"], ["auto1", "Automatic · one source"]],
  interception: [["ball", "Ball at contact (3D fit)"], ["proxy", "Opponent's feet (position proxy)"]],
  origin: [["ball", "Ball at contact (3D fit)"], ["proxy", "Hitter's feet (position proxy)"]],
};
const STROKES = [["forehand", "Forehand"], ["backhand", "Backhand"]];
const INCOMING = [["after bounce", "After a bounce"], ["volley", "Volley (out of the air)"], ["serve", "Serve (no incoming ball)"]];
const STATUSES = [["bounced", "Bounced first"], ["intercepted", "Intercepted"], ["net", "Ended at the net"], ["unresolved", "Unresolved"]];

const cap = s => s ? s[0].toUpperCase() + s.slice(1) : "";
const pct = (a, b) => b ? Math.round(100 * a / b) : null;
const fmtTime = t => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };
const median = a => { if (!a.length) return null; const b = [...a].sort((x, y) => x - y), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
const quantile = (a, q) => { if (!a.length) return null; const b = [...a].sort((x, y) => x - y), i = (b.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i); return b[lo] + (b[hi] - b[lo]) * (i - lo); };

function zoneLabel(z) {
  if (!z) return "";
  if (z === "boundary") return "On a zone boundary";
  if (z.startsWith("band:")) { const d = z.slice(5); return d === "behind" ? "Behind the baseline (all)" : `${cap(d)} (whole row)`; }
  const [d, w] = z.split(" ");
  if (w && w.startsWith("wide")) return `Wide of the ${w.slice(5)} sideline (${d === "behind" ? "behind the baseline" : d})`;
  if (d === "behind") return `Behind the baseline, ${w}`;
  return cap(z);
}

// ---------- who: a side of the match (singles player / doubles team) or a named doubles player ----------
function sidesOf(R) { return Object.keys(R.players || {}); }
function teamOf(R, name) { return Object.entries(R.teams || {}).find(([, ns]) => ns.includes(name))?.[0] || null; }
function selectionOptions(R) {
  // [{key, label, kind: 'side'|'player', side, team}] - doubles: both teams, then their named players
  const sides = sidesOf(R), out = sides.map((n, i) => ({ key: n, label: n, kind: "side", side: n, idx: i, team: R.type === "doubles" }));
  if (R.type === "doubles" && R.teams) for (const [t, names] of Object.entries(R.teams))
    for (const n of names) if ((R.individuals || {})[n]) out.push({ key: n, label: n, kind: "player", side: t, idx: sides.indexOf(t), team: false });
  return out;
}
function resolveSelection(R, key) {
  const opts = selectionOptions(R);
  return opts.find(o => o.key === key) || opts[0];
}

let _fps = new WeakMap();
function fpsOf(R) {
  if (_fps.has(R)) return _fps.get(R);
  const r = R.rallies.flatMap(x => x.shots).filter(s => s.f > 0 && s.t > 0).map(s => s.f / s.t);
  const f = median(r) || 30;
  _fps.set(R, f);
  return f;
}

// Shots credited to the selection that carry a placement record. Inferred shots (deduced from the rules, not seen)
// carry no measurement and are left out, as in the reports.
function shotsFor(R, sel) {
  const out = [];
  for (const r of R.rallies) for (const s of r.shots) {
    if (!s.pl || s.inferred) continue;
    if (sel.kind === "player" ? s.hitter !== sel.key : s.player !== sel.side) continue;
    out.push(Object.assign({ rid: r.id }, s));
  }
  return out;
}

function locOf(s, view) {
  const l = s.pl[VIEWS[view].key];
  if (!l) return null;
  if (view === "origin") return l.zone ? l : null;
  return l.state === "observed" ? l : null;
}
function evidenceOf(s, view) {
  const l = locOf(s, view);
  if (!l) return null;
  if (view === "landing") return l.quality === "human-reviewed" ? "human" : l.quality === "two automatic sources agree" ? "auto2" : "auto1";
  return /proxy/.test(l.evidence || "") ? "proxy" : "ball";
}
const EV_BADGE = { human: ["Human-reviewed", "human"], auto2: ["Automatically detected", "auto"], auto1: ["Automatically detected", "auto"],
  ball: ["Automatically detected", "auto"], proxy: ["Position proxy", "proxy"] };

// ---------- filters ----------
const DEFAULT_FILTERS = { role: "all", stroke: "", incoming: "", status: "", ev: "" };
// status applies only to "Hit from" (the other two views ARE a status); evidence options are per view.
function applicable(key, view) {
  if (key === "status") return view === "origin";
  return true;
}
function applyFilters(shots, f, view) {
  return shots.filter(s => (!f.role || f.role === "all" || s.pl.role === f.role)
    && (!f.stroke || s.stroke === f.stroke)
    && (!f.incoming || s.pl.incoming === f.incoming)
    && (!f.status || !applicable("status", view) || s.pl.status === f.status)
    && (!f.ev || evidenceOf(s, view) === f.ev));
}

// ---------- aggregation for one view ----------
// Denominator of every zone share = all recorded locations of this view (boundary, outside and other-half ones
// included), so zone shares + the "not in a zone" groups add up to 100%.
function aggregate(shots, view) {
  const V = VIEWS[view], cells = {}, boundary = [], outside = [], otherHalf = [], recorded = [], missing = [], status = {};
  let population = 0;
  for (const s of shots) {
    status[s.pl.status] = (status[s.pl.status] || 0) + 1;
    if (!V.population(s)) continue;
    population++;
    const l = locOf(s, view);
    if (!l) { missing.push(s); continue; }
    recorded.push(s);
    if (l.half !== V.half) otherHalf.push(s);
    else if (l.zone === "boundary") boundary.push(s);
    else if (isCell(l.zone, view)) (cells[l.zone] = cells[l.zone] || []).push(s);
    else outside.push(s);
  }
  return { view, matching: shots.length, status, population, recorded, cells, boundary, outside, otherHalf, missing, denominator: recorded.length };
}
// the shots behind a selection key: a cell zone, a whole row ('band:deep'), or a not-in-a-zone group
function selectShots(agg, zone) {
  if (!zone) return agg.recorded;
  if (zone === "boundary") return agg.boundary;
  if (zone === "outside") return agg.outside;
  if (zone === "otherhalf") return agg.otherHalf;
  if (zone.startsWith("band:")) { const d = zone.slice(5); return WIDTHS.flatMap(w => agg.cells[`${d} ${w}`] || []).sort((a, b) => a.t - b.t); }
  return agg.cells[zone] || [];
}
function validZone(zone, view) {
  if (!zone) return true;
  if (["boundary", "outside", "otherhalf"].includes(zone)) return true;
  if (zone.startsWith("band:")) return ROWS[view].includes(zone.slice(5));
  return isCell(zone, view);
}

// ---------- estimated speed (one definition: physics launch speed off the paddle; whole mph) ----------
function speedStats(shots) {
  const v = shots.map(s => s.pl.speed && s.pl.speed.mph).filter(x => x != null);
  return { n: shots.length, usable: v.length, median: v.length ? Math.round(median(v)) : null,
    q1: v.length >= 6 ? Math.round(quantile(v, 0.25)) : null, q3: v.length >= 6 ? Math.round(quantile(v, 0.75)) : null,
    max: v.length ? Math.round(Math.max(...v)) : null };
}

// ---------- clips ----------
const LEAD_S = 1.8, FOLLOW_S = 1.3, MAX_CLIP_S = 7;
// Video-time window for one shot: lead-in before the contact, follow-through past the shot's first event (landing,
// interception, or the next contact). Report times are feed times; R.video_offset maps them onto a cut video.
function clipWindow(R, s, view) {
  const fps = fpsOf(R), off = R.video_offset || 0;
  const l = view ? locOf(s, view) : null;
  const evF = (l && l.frame) ?? (s.pl.landing && s.pl.landing.frame) ?? (s.pl.interception && s.pl.interception.frame) ?? s.pl.end_frame;
  const evT = evF != null && s.f != null ? s.t + (evF - s.f) / fps : null;
  const start = s.t - LEAD_S, end = Math.min(s.t + MAX_CLIP_S, Math.max(s.t + 1, (evT ?? s.t + 1) + FOLLOW_S));
  return { start: Math.max(0, start - off), end: end - off, hit: s.t - off, event: evT == null ? null : evT - off };
}
function rallyWindow(R, r) {
  const off = R.video_offset || 0;
  return { start: Math.max(0, r.start_t - 1 - off), end: r.end_t + 1.5 - off, hit: r.start_t - off, event: null };
}
function shotLine(s, view) {   // "Third · FH · midcourt left → deep left"
  const o = s.pl.origin, l = view === "interception" ? s.pl.interception : s.pl.landing;
  const zn = z => z.zone === "boundary" ? `zone boundary (${(z.candidates || []).slice(0, 3).join(" / ")})` : z.zone;
  const ev = s.pl.status === "bounced" ? (l && l.state === "observed" ? zn(l) : "bounce position unknown")
    : s.pl.status === "intercepted" ? (s.pl.interception && s.pl.interception.state === "observed" ? `intercepted at ${zn(s.pl.interception)}` : "intercepted")
    : s.pl.status === "net" ? "net" : "outcome unresolved";
  return { role: ROLE_SHORT[s.pl.role] || "", stroke: s.stroke ? (s.stroke === "forehand" ? "Forehand" : "Backhand") : "",
    path: `${o && o.zone ? (o.zone === "boundary" ? "zone boundary" : o.zone) : "?"} → ${ev}`, mph: s.pl.speed && s.pl.speed.mph != null ? Math.round(s.pl.speed.mph) : null };
}

// ---------- URL state ----------
const SECTIONS = ["overview", "placement", "movement", "partnership", "clips", "advanced"];
const STATE_DEFAULTS = { sec: "overview", p: "", role: "all", view: "landing", mode: "zones", zone: "", stroke: "", incoming: "", status: "", ev: "", list: "", clip: "" };
function parseState(search, hash) {
  const q = new URLSearchParams(search || ""), st = { m: q.get("m") || "" };
  for (const [k, d] of Object.entries(STATE_DEFAULTS)) st[k] = q.get(k) ?? d;
  // links from before the redesign: #players / #rallies / #replay / #accuracy
  const h = (hash || "").replace(/^#/, "");
  const legacy = { partnership: "partnership", players: "placement", overview: "overview", rallies: "clips", replay: "advanced", placement: "placement", movement: "movement", clips: "clips", advanced: "advanced" };
  if (!q.get("sec") && legacy[h]) st.sec = legacy[h];
  if (!q.get("sec") && q.get("tab") && legacy[q.get("tab")]) st.sec = legacy[q.get("tab")];
  if (h === "rallies" && !q.get("list")) st.list = "rallies";
  if (!SECTIONS.includes(st.sec)) st.sec = "overview";
  if (!ROLES.some(([k]) => k === st.role)) st.role = "all";
  if (!VIEWS[st.view]) st.view = "landing";
  if (!["zones", "shots"].includes(st.mode)) st.mode = "zones";
  if (st.status && !applicable("status", st.view)) st.status = "";
  if (st.ev && !EVIDENCE[st.view].some(([k]) => k === st.ev)) st.ev = "";
  if (!validZone(st.zone, st.view)) st.zone = "";
  return st;
}
function stateToSearch(st) {
  const q = new URLSearchParams();
  if (st.m) q.set("m", st.m);
  for (const [k, d] of Object.entries(STATE_DEFAULTS)) if (st[k] != null && st[k] !== "" && st[k] !== d) q.set(k, st[k]);
  return "?" + q.toString();
}
// switching the view keeps the situation filters but drops what would mean something else in the new view
function switchView(st, view) {
  const notes = [];
  const n = Object.assign({}, st, { view });
  if (st.zone) { n.zone = ""; }
  if (st.ev) { n.ev = ""; notes.push("Evidence filter cleared - each map view has its own evidence types."); }
  if (st.status && !applicable("status", view)) { n.status = ""; notes.push(`Outgoing-flight filter cleared - it applies only to the Hit from view (${VIEWS[view].label} already shows ${VIEWS[view].popNoun}).`); }
  return { state: n, notes };
}

// ---------- automatic observations ----------
// Descriptive tendencies only (never advice), each with its denominator, scope and quality, and only when the
// sample clears fixed minimums. Fewer cards are returned when fewer are supported.
const OBS_MIN = { landingAll: 15, landingRole: 10, topCount: 5, topShare: 0.25, lead: 2, rowShare: 0.6, interceptResolved: 10, interceptCount: 4, interceptShare: 0.35, originN: 15, originShare: 0.4 };
function observations(R, sel, opts = {}) {
  const max = opts.max ?? 3, shots = opts.shots || shotsFor(R, sel), who = opts.who || sel.label || sel.key;
  const scope = opts.scope || "this match", cand = [];
  for (const role of ["all", "serve", "return", "third", "later"]) {
    const a = aggregate(applyFilters(shots, { role }, "landing"), "landing"), n = a.denominator;
    if (n < (role === "all" ? OBS_MIN.landingAll : OBS_MIN.landingRole)) continue;
    const top = Object.entries(a.cells).map(([z, l]) => [z, l.length]).sort((x, y) => y[1] - x[1]);
    if (!top.length) continue;
    const [z, c] = top[0], c2 = top[1] ? top[1][1] : 0;
    const human = a.recorded.filter(s => evidenceOf(s, "landing") === "human").length;
    const lq = [notInZone(a, "landings"), human ? `${human} of ${n} landings human-reviewed, the rest automatically detected` : "All landings automatically detected"].filter(Boolean);
    // a whole depth row (deep / midcourt / kitchen) - robust when two neighbouring zones are close
    const rows = DEPTHS.map(d => [d, WIDTHS.reduce((x, w) => x + ((a.cells[`${d} ${w}`] || []).length), 0)]).sort((x, y) => y[1] - x[1]);
    if (rows[0][1] / n >= OBS_MIN.rowShare && rows[0][1] >= OBS_MIN.topCount) {
      const [d, rc] = rows[0], dn = d === "deep" ? "deep (14.5–22 ft from the net)" : d === "kitchen" ? "in the kitchen (within 7 ft of the net)" : "in midcourt (7–14.5 ft from the net)";
      cand.push({ id: `landingrow:${role}:${d}`, kind: "landing", group: `landing:${role}`, role, zone: `band:${d}`, count: rc, n, score: (rc / n) * Math.min(1, n / 30) * 0.8,
        headline: role === "all" ? `Most observed landings were ${dn}.` : `Most ${ROLE_NOUN[role]} landed ${dn}.`,
        measure: `${rc} of ${n} ${role === "all" ? "recorded landings" : `recorded ${ROLE_NOUN[role].replace(/s$/, "")} landings`} (${pct(rc, n)}%)`,
        scope: `${who} · ${role === "all" ? "all shots" : ROLE_NOUN[role]} · observed first bounce · ${scope}`,
        quality: lq, evidence: human === n ? "human" : "auto", limited: n < 20,
        link: { sec: "placement", role, view: "landing", zone: `band:${d}`, mode: "zones" } });
    }
    // one zone that clearly leads (the runner-up is not close)
    if (c < OBS_MIN.topCount || c / n < OBS_MIN.topShare || c - c2 < Math.max(OBS_MIN.lead, Math.ceil(0.2 * c))) continue;
    cand.push({ id: `landing:${role}:${z}`, kind: "landing", group: `landing:${role}`, role, zone: z, count: c, n, score: (c / n) * Math.min(1, n / 30) * 1.6,
      headline: role === "all" ? `${zoneLabel(z)} was the most common observed landing zone.` : `${cap(ROLE_NOUN[role])} most often landed ${z}.`,
      measure: `${c} of ${n} ${role === "all" ? "recorded landings" : `recorded ${ROLE_NOUN[role].replace(/s$/, "")} landings`} (${pct(c, n)}%)`,
      scope: `${who} · ${role === "all" ? "all shots" : ROLE_NOUN[role]} · observed first bounce · ${scope}`,
      quality: lq, evidence: human === n ? "human" : "auto", limited: n < 20,
      link: { sec: "placement", role, view: "landing", zone: z, mode: "zones" } });
  }
  for (const role of ["third", "later", "all"]) {
    const S = applyFilters(shots, { role }, "origin"), st = aggregate(S, "origin").status;
    const resolved = (st.bounced || 0) + (st.intercepted || 0) + (st.net || 0), i = st.intercepted || 0;
    if (resolved < OBS_MIN.interceptResolved || i < OBS_MIN.interceptCount || i / resolved < OBS_MIN.interceptShare) continue;
    cand.push({ id: `intercepted:${role}`, kind: "intercepted", role, count: i, n: resolved, score: (i / resolved) * Math.min(1, resolved / 30) * 0.95,
      headline: `${cap(role === "all" ? "shots" : ROLE_NOUN[role])} were often intercepted before they bounced.`,
      measure: `${i} of ${resolved} ${role === "all" ? "shots" : ROLE_NOUN[role]} with a known outcome were intercepted (${pct(i, resolved)}%)`,
      scope: `${who} · ${role === "all" ? "all shots" : ROLE_NOUN[role]} · outgoing-flight status · ${scope}`,
      quality: [st.unresolved ? `${st.unresolved} unresolved ${st.unresolved === 1 ? "shot" : "shots"} not counted` : "",
        "An interception needs a volley flag and dense ball tracking, so some may be missed"].filter(Boolean),
      evidence: "auto", limited: resolved < 20,
      link: { sec: "placement", role, view: "origin", status: "intercepted", zone: "", mode: "zones" } });
    break;   // one interception observation at most
  }
  {
    const a = aggregate(applyFilters(shots, { role: "later" }, "origin"), "origin"), n = a.denominator;
    const bands = ROWS.origin.map(d => [d, WIDTHS.reduce((x, w) => x + ((a.cells[`${d} ${w}`] || []).length), 0)]).sort((x, y) => y[1] - x[1]);
    if (n >= OBS_MIN.originN && bands[0][1] / n >= OBS_MIN.originShare) {
      const [d, c] = bands[0], proxy = a.recorded.filter(s => evidenceOf(s, "origin") === "proxy").length;
      cand.push({ id: `origin:later:${d}`, kind: "origin", role: "later", zone: `band:${d}`, count: c, n, score: (c / n) * Math.min(1, n / 30) * 0.9,
        headline: d === "behind" ? "Later rally shots were most often hit from behind the baseline." : `Later rally shots were most often hit from the ${d === "deep" ? "deep court" : d}.`,
        measure: `${c} of ${n} recorded contact points (${pct(c, n)}%)`,
        scope: `${who} · later rally shots · where the shot was hit · ${scope}`,
        quality: [notInZone(a, "contact points"), proxy ? `${proxy} of ${n} positions are the hitter's feet (position proxy), the rest the ball at contact` : ""].filter(Boolean),
        evidence: proxy ? "proxy" : "auto", limited: n < 20,
        link: { sec: "placement", role: "later", view: "origin", zone: `band:${d}`, mode: "zones" } });
    }
  }
  const out = [], used = new Set(), kinds = {};
  for (const c of cand.sort((x, y) => y.score - x.score)) {
    if (out.length >= max) break;
    if (c.group && used.has(c.group)) continue;                     // one landing statement per situation
    if (c.kind === "landing" && used.has(c.zone)) continue;         // the same zone twice says the same thing
    if ((kinds[c.kind] || 0) >= 2) continue;
    out.push(c); kinds[c.kind] = (kinds[c.kind] || 0) + 1;
    if (c.group) used.add(c.group); if (c.kind === "landing") used.add(c.zone);
  }
  return out;
}
function notInZone(a, noun) {
  const parts = [];
  if (a.boundary.length) parts.push(`${a.boundary.length} on a zone boundary`);
  if (a.outside.length) parts.push(`${a.outside.length} estimated outside the court`);
  if (a.otherHalf.length) parts.push(`${a.otherHalf.length} on the other half`);
  return parts.length ? `${parts.join(", ")} - counted in the ${a.denominator} but in no zone` : "";
}

// Doubles: replace the report's hitter with the attribution from outputs/reports/<report>.doubles.json
// (pickleball/doubles_players.attribute_hitters). Unattributed shots get hitter null: team-level only.
function applyHitters(R, D) {
  if (!D || !D.hitters) return 0;
  let n = 0;
  for (const r of R.rallies) for (const s of r.shots) {
    const h = D.hitters[`${r.id}:${s.i}`];
    if (!h) continue;
    s.hitter_report = s.hitter; s.hitter = h.name; s.hitter_method = h.method; n++;
  }
  return n;
}

// placement validation per venue (filled from accuracy.js in the browser)
function venueValidation(tournament) {
  const V = (root.PLACEMENT_VALIDATION || {});
  return V[tournament] || V._default || null;
}

const API = { ROLES, ROLE_NOUN, ROLE_SHORT, STATUS_LABEL, VIEWS, DEPTHS, WIDTHS, ROWS, BANDS, EVIDENCE, EV_BADGE, STROKES, INCOMING, STATUSES,
  SECTIONS, STATE_DEFAULTS, DEFAULT_FILTERS, OBS_MIN, LEAD_S, FOLLOW_S,
  cap, pct, fmtTime, median, zoneLabel, sidesOf, teamOf, selectionOptions, resolveSelection, fpsOf, shotsFor, locOf, evidenceOf,
  cellZones, isCell, applicable, applyFilters, aggregate, selectShots, validZone, speedStats, clipWindow, rallyWindow, shotLine,
  parseState, stateToSearch, switchView, observations, notInZone, venueValidation, applyHitters };
if (typeof module !== "undefined" && module.exports) module.exports = API;
root.PB = API;
})(typeof window !== "undefined" ? window : globalThis);
