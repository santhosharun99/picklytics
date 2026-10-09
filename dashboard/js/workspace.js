// Match workspace (hub.html?m=<match id>): Overview · Placement & speed · Movement · Clips · Advanced analysis.
// All view state (section, player/team, situation, map view, zone, filters, clip) lives in the URL, so back/forward
// and bookmarks restore it. Numbers come from the cached report (outputs/reports/<report>.json); counting rules live
// in js/coach-core.js. One video player is moved between sections, so playback and the playlist survive navigation.
(function () {
"use strict";
const PB = window.PB, $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const { cap, pct, fmtTime, zoneLabel, VIEWS, ROLES, ROLE_NOUN } = PB;
const getJSON = async u => { const r = await fetch(u, { cache: "no-store" }); if (!r.ok) throw new Error(`${r.status} ${u}`); return r.json(); };
const pslug = n => n.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const NAV = [["overview", "Overview"], ["placement", "Placement & speed"], ["movement", "Movement"], ["clips", "Clips"]];
const VIEW_PHRASE = { landing: "observed landings", interception: "interceptions", origin: "hit from" };
let J, M, T, R = null, A = null, DB = null, PIDS = new Set(), ST, SEL = null, FLASH = [], reportError = false, RNO = new Map();
const rno = id => RNO.get(id) ?? id;   // rally number = position in the match (report ids skip removed replays)

// ------------------------------------------------------------------ boot
async function init() {
  try { J = await getJSON("matches.json"); } catch (e) { return fatal("The match list (dashboard/matches.json) could not be loaded."); }
  ST = PB.parseState(location.search, location.hash);
  const q = new URLSearchParams(location.search), hash = location.hash.replace(/^#/, "");
  M = J.matches.find(m => m.id === ST.m);
  if (!M) {
    if (ST.m) return fatal(`There is no match with the id “${esc(ST.m)}”. <a href="matches.html">Back to all matches</a>`);
    return siteAccuracy();
  }
  if (hash === "scout") { location.replace(`scout.html?m=${encodeURIComponent(M.id)}`); return; }
  T = (J.tournaments || []).find(t => t.id === M.tournament);
  try { PIDS = new Set((await getJSON("players.json")).players.map(p => p.id)); } catch (e) { /* player links are optional */ }
  if (M.report) { try { R = await getJSON(`../outputs/reports/${M.report}.json`); RNO = new Map(R.rallies.map((r, i) => [r.id, i + 1])); } catch (e) { reportError = true; } }
  if (R && R.type === "doubles") {   // individual + partnership numbers (training/build_doubles_players.py)
    try { DB = await getJSON(`../outputs/reports/${M.report}.doubles.json`); PB.applyHitters(R, DB); } catch (e) { DB = null; }
  }
  else if (M.run) { try { A = await getJSON(`../outputs/${M.run}/analysis.json`); } catch (e) { /* status view copes */ } }
  if (hash || q.has("tab")) history.replaceState(null, "", PB.stateToSearch(ST));
  document.title = `${M.title} · Picklytics`;
  $("#loading")?.remove();
  renderHeader();
  $("#wsbar").hidden = false;
  $("#covBtn").onclick = openDQ;
  window.addEventListener("popstate", () => { ST = PB.parseState(location.search, ""); FLASH = []; render(); });
  matchMedia("(max-width: 520px)").addEventListener("change", () => ST.sec === "placement" && render());
  render();
  if (hash === "accuracy") openDQ();
}
function fatal(html) {
  $("#loading")?.remove();
  $("#main").insertAdjacentHTML("afterbegin", `<div class="state-empty" role="alert">${html}</div>`);
}

// state change: push a history entry (section / who / filters / view / zone), then re-render
function go(patch, { replace = false, keepFlash = false } = {}) {
  const prevSec = ST.sec;
  ST = Object.assign({}, ST, patch);
  if (!("clip" in patch)) ST.clip = "";
  const url = PB.stateToSearch(ST);
  if (url !== location.search) history[replace ? "replaceState" : "pushState"](null, "", url);
  if (!keepFlash) FLASH = [];
  render();
  if (ST.sec !== prevSec) { const top = $("#wsbar").getBoundingClientRect().top + scrollY - 4; if (scrollY > top) scrollTo({ top, behavior: "auto" }); }
}
const placementPatch = p => p;   // (readability) every placement change keeps the other state

// ------------------------------------------------------------------ header + bar
function plink(n, team) {
  const id = [team && pslug(`${n} ${team}`), pslug(n)].find(x => x && PIDS.has(x));
  return id ? `<a href="player.html?p=${id}">${esc(n)}</a>` : esc(n);
}
function games() {
  const g = {}, rally = R.rules && R.rules.scoring === "rally";
  for (const r of R.rallies) if (r.game && r.winner) {
    g[r.game] = g[r.game] || {}; if (rally || r.server === r.winner) g[r.game][r.winner] = (g[r.game][r.winner] || 0) + 1;
  }
  return g;
}
function overlayCheck() {   // agreement of tracked rally winners with the broadcast score overlay, where it settles them
  const c = R.rallies.map(r => [r, r.scoreboard || {}]).filter(([, b]) => b.winner);
  return { n: c.length, ok: c.filter(([r, b]) => r.winner === b.winner).length };
}
// Rally-by-rally winners are summarised only for singles, and only where no cross-check says they are poor.
function winnersReliable() {
  if (!R || R.type === "doubles") return false;
  const c = overlayCheck();
  return !(c.n >= 5 && c.ok / c.n < 0.85);
}
function renderHeader() {
  const crumb = [`<a href="matches.html">Tournaments</a>`, T ? `<a href="matches.html?t=${encodeURIComponent(T.id)}">${esc(T.name)}</a>` : "", M.group ? `<span>${esc(M.group)}</span>` : "", `<span class="here" aria-current="page">${esc(M.title)}</span>`].filter(Boolean);
  $("#crumb").innerHTML = crumb.join(`<span aria-hidden="true">›</span>`);
  const sides = R ? PB.sidesOf(R) : M.players;
  const kind = M.type === "doubles" ? "Doubles" : (M.round || "").toLowerCase().includes("team") ? "Team singles" : "Singles";
  const who = sides.map((n, i) => `<span><span class="dot" style="background:var(--series-${i + 1})" aria-hidden="true"></span>${R && R.teams && R.teams[n] ? `<span><b>${esc(n)}</b> <span>(${R.teams[n].map(x => plink(x, n)).join(", ")})</span></span>` : plink(n)}</span>`).join("");
  const official = (R && R.official) || M.result;
  let res = "";
  if (official) res += `<div><span class="badge official">Official result</span> <span class="off">${esc(official)}</span></div>`;
  if (R && R.type !== "doubles") {
    const g = games(), names = PB.sidesOf(R);
    const txt = Object.entries(g).map(([k, v]) => `G${k} ${v[names[0]] || 0}–${v[names[1]] || 0}`).join(" · ");
    if (txt) res += `<div class="rc"><span class="badge recon" title="Rebuilt from the tracked server and rally winners; ${esc(M.check || "")}">Reconstructed from tracking</span> ${txt} <span class="note">(${esc(names[0])} first)</span></div>`;
  }
  $("#mh").innerHTML = `<div style="min-width:0"><div class="ev"><span class="fmt">${kind}</span>${esc(M.event)} · ${esc(M.round)}</div>
    <h1>${esc(M.title)}</h1><div class="who-line">${who}</div></div>${res ? `<div class="mh-res">${res}</div>` : ""}`;
}
function renderBar() {
  const has = !!R;
  const nav = DB ? [...NAV.slice(0, 3), ["partnership", "Partnership"], NAV[3]] : NAV;
  $("#secnav").innerHTML = nav.map(([k, l]) => `<a href="${PB.stateToSearch(Object.assign({}, ST, { sec: k, clip: "" }))}" data-sec="${k}"${ST.sec === k ? ' aria-current="page"' : ""}${!has && k !== "overview" ? ' aria-disabled="true" tabindex="-1" title="No match report for this match yet"' : ""}>${l}</a>`).join("")
    + `<span class="sep" aria-hidden="true"></span><a class="adv" href="${PB.stateToSearch(Object.assign({}, ST, { sec: "advanced", clip: "" }))}" data-sec="advanced"${ST.sec === "advanced" ? ' aria-current="page"' : ""}${!has ? ' aria-disabled="true" tabindex="-1"' : ""}>Advanced analysis</a>`;
  $$("#secnav a").forEach(a => a.onclick = e => { if (e.metaKey || e.ctrlKey || e.shiftKey) return; e.preventDefault(); go({ sec: a.dataset.sec }); });
  if (!has) { $("#who").innerHTML = ""; $("#covBtn").hidden = true; $("#briefBtn").hidden = true; return; }
  const opts = PB.selectionOptions(R);
  if (opts.length <= 2 && opts.every(o => o.kind === "side")) {
    $("#who").innerHTML = `<span class="sr" id="whoLbl">Analyse</span><div class="seg" role="group" aria-labelledby="whoLbl">${opts.map(o =>
      `<button type="button" data-k="${esc(o.key)}" aria-pressed="${o.key === SEL.key}"><span class="dot" style="background:var(--series-${o.idx + 1})" aria-hidden="true"></span>${esc(o.label)}</button>`).join("")}</div>`;
    $$("#who button").forEach(b => b.onclick = () => go({ p: b.dataset.k, zone: "" }));
  } else {
    const groups = PB.sidesOf(R).map(side => {
      const inner = opts.filter(o => o.side === side);
      return `<optgroup label="${esc(side)}">${inner.map(o => `<option value="${esc(o.key)}"${o.key === SEL.key ? " selected" : ""}>${o.kind === "side" ? `${esc(side)} · whole team` : `${esc(o.label)} · individual`}</option>`).join("")}</optgroup>`;
    }).join("");
    $("#who").innerHTML = `<label for="whoSel">Analyse</label><select id="whoSel">${groups}</select>`;
    $("#whoSel").onchange = e => go({ p: e.target.value, zone: "" });
  }
  const sh = PB.shotsFor(R, SEL), land = PB.aggregate(sh, "landing").denominator, sp = PB.speedStats(sh);
  $("#covBtn").hidden = false;
  $("#covBtn").innerHTML = `Data quality · <b>${sh.length}</b> shots · <b>${land}</b> landings`;
  $("#covBtn").title = `Estimated speed usable for ${sp.usable} of ${sh.length} shots. Open for coverage, evidence and validation.`;
  $("#covBtn").setAttribute("aria-label", `Data quality and coverage for ${SEL.label}: ${sh.length} analysed shots, ${land} recorded landings, estimated speed for ${sp.usable} of ${sh.length}`);
  $("#briefBtn").hidden = false;
  $("#briefBtn").href = `scout.html?m=${encodeURIComponent(M.id)}&p=${encodeURIComponent(SEL.key)}`;
}

function render() {
  if (R) { SEL = PB.resolveSelection(R, ST.p); SEL.label = SEL.kind === "side" && R.type === "doubles" ? `${SEL.key} (team)` : SEL.key; }
  if (!R && ST.sec !== "overview") ST.sec = "overview";
  renderBar();
  if (ST.sec === "partnership" && !DB) ST.sec = "overview";
  for (const s of ["overview", "placement", "movement", "partnership", "clips", "advanced", "accuracy"]) $(`#sec-${s}`).hidden = s !== ST.sec;
  ({ overview: R ? renderOverview : renderStatus, placement: renderPlacement, movement: renderMovement, partnership: renderPartnership, clips: renderClips, advanced: renderAdvanced })[ST.sec]();
}

// ------------------------------------------------------------------ shared helpers
const whoNoun = () => SEL.kind === "side" && R.type === "doubles" ? `${SEL.key} (both players)` : SEL.key;
function evBadge(ev, view) {
  if (!ev) return "";
  const [t, c] = PB.EV_BADGE[ev];
  const tip = view === "landing" ? { human: "Bounce position tagged by a person", auto2: "Physics fit and bounce detector agree", auto1: "One automatic source" }[ev]
    : ev === "proxy" ? "Player's feet used as the position (no ball fit at contact)" : "Ball position at contact from the 3D fit";
  return `<span class="badge ${c}" title="${esc(tip)}">${t}</span>`;
}
function speedText(sp, noun = "shots") {
  if (!sp.n) return "";
  if (!sp.usable) return `<span>Estimated speed: <b>unknown</b></span><span>no usable estimate for these ${sp.n} ${noun}</span>`;
  return `<span>Estimated speed · median <b>${sp.median} mph</b></span>${sp.q1 != null ? `<span>middle half ${sp.q1}–${sp.q3} mph</span>` : ""}<span>usable for ${sp.usable} of ${sp.n} ${noun}</span>`;
}
function curFilters() { return { role: ST.role, stroke: ST.stroke, incoming: ST.incoming, status: ST.status, ev: ST.ev }; }
function selectionModel() {
  const all = PB.shotsFor(R, SEL), shots = PB.applyFilters(all, curFilters(), ST.view), agg = PB.aggregate(shots, ST.view);
  const list = PB.selectShots(agg, ST.zone);
  return { all, shots, agg, list };
}
function selectionLabel() {
  const z = ST.zone ? (ST.zone === "outside" ? "Outside the court" : ST.zone === "otherhalf" ? "On the other half" : zoneLabel(ST.zone)) : "All zones";
  const extra = [ST.stroke && cap(ST.stroke), ST.incoming && PB.INCOMING.find(x => x[0] === ST.incoming)?.[1].toLowerCase(), ST.status && PB.applicable("status", ST.view) && PB.STATUS_LABEL[ST.status], ST.ev && PB.EVIDENCE[ST.view].find(x => x[0] === ST.ev)?.[1].toLowerCase()].filter(Boolean);
  return `${z} · ${VIEW_PHRASE[ST.view]} · ${ST.role === "all" ? "all shots" : ROLE_NOUN[ST.role]}${extra.length ? " · " + extra.join(" · ") : ""}`;
}
const selKey = () => ["sel", SEL.key, ST.view, ST.role, ST.zone, ST.stroke, ST.incoming, ST.status, ST.ev].join("|");
const shotItems = (list, view) => list.map(s => ({ kind: "shot", s, view }));
const rallyItems = ids => ids.map(id => R.rallies.find(r => r.id === id)).filter(Boolean).map(r => ({ kind: "rally", r }));
const momentItems = ms => ms.map(m => ({ kind: "moment", r: R.rallies.find(r => r.id === m.rally), t: m.t, mph: m.mph })).filter(x => x.r);
function watchCustom(slot, items, title, key) {
  if (!items.length) return;
  Player.mount(slot); slot.hidden = false;
  Player.load(items, { key, title, list: title });
  Player.play(0);
  slot.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ------------------------------------------------------------------ video player (one instance, moved between slots)
const Player = (() => {
  let root, video, items = [], idx = -1, key = "", title = "", stopAt = null, srcSet = false;
  function build() {
    if (root) return root;
    root = document.createElement("div");
    root.className = "player";
    root.innerHTML = `<div class="now"><span class="t" id="plTitle"></span><span class="pos" id="plPos" aria-live="polite"></span></div>
      <div class="vwrap"><video id="vid" preload="none" playsinline controls></video><div class="vmsg" id="vmsg" hidden></div></div>
      <div class="ctrl"><button type="button" class="btn sm" id="plPrev" aria-label="Previous clip">◀ Previous</button>
        <button type="button" class="btn sm" id="plReplay" aria-label="Replay this clip">↺ Replay</button>
        <button type="button" class="btn sm" id="plNext" aria-label="Next clip">Next ▶</button>
        <button type="button" class="btn sm ghost" id="plBack" hidden>↑ Back to court</button>
        <label><input type="checkbox" id="plAuto"> Play all in order</label></div>
      <div class="note" id="plDesc"></div>
      <ol class="playlist" id="plList"></ol>`;
    video = $("video", root);
    if (M.thumb) video.poster = M.thumb;
    if (!(R && R.video) && !M.video) showMsg("No video file is linked to this match.");
    video.addEventListener("error", () => showMsg(`The match video could not be loaded (${esc((R && R.video) || M.video)}). Clips need the video file on this computer.`));
    video.addEventListener("timeupdate", () => {
      if (stopAt == null || video.currentTime < stopAt) return;
      stopAt = null;
      if ($("#plAuto", root).checked && idx < items.length - 1) play(idx + 1); else video.pause();
    });
    $("#plPrev", root).onclick = () => items.length && play(Math.max(0, idx - 1));
    $("#plNext", root).onclick = () => items.length && play(Math.min(items.length - 1, idx + 1));
    $("#plReplay", root).onclick = () => items.length && play(Math.max(0, idx));
    $("#plBack", root).onclick = () => { const c = $("#courtPanel"); if (c) c.scrollIntoView({ behavior: "smooth", block: "start" }); };
    return root;
  }
  function showMsg(t) { const m = $("#vmsg", root); m.innerHTML = t; m.hidden = false; }
  function win(it) {
    if (it.kind === "shot") return PB.clipWindow(R, it.s, it.view);
    if (it.kind === "moment") { const off = R.video_offset || 0; return { start: Math.max(0, it.t - 2 - off), end: it.t + 3 - off, hit: it.t - off }; }
    return PB.rallyWindow(R, it.r);
  }
  function itemHTML(it, i) {
    const w = win(it);
    if (it.kind === "shot") {
      const s = it.s, L = PB.shotLine(s, it.view), ev = PB.evidenceOf(s, it.view);
      return `<span class="i">${i + 1}</span><span class="a">Rally ${rno(s.rid)} · ${fmtTime(w.hit)} · ${L.role}${L.stroke ? " · " + L.stroke : ""}</span>
        <span class="c">${L.mph != null ? `${L.mph} mph<small>estimated</small>` : `–<small>no speed</small>`}</span>
        <span class="b">${esc(L.path)} ${evBadge(ev, it.view)}</span>`;
    }
    const r = it.r, dur = Math.round(r.end_t - r.start_t);
    if (it.kind === "moment") return `<span class="i">${i + 1}</span><span class="a">Rally ${rno(r.id)} · ${fmtTime(w.hit)}</span><span class="c">${it.mph ? `${it.mph} mph<small>running</small>` : ""}</span><span class="b">from 2 s before the moment</span>`;
    return `<span class="i">${i + 1}</span><span class="a">Rally ${rno(r.id)} · ${fmtTime(w.hit)}${r.game ? ` · G${r.game} ${esc(r.score_before ?? "")}` : ""}</span>
      <span class="c">${r.n_shots} shots<small>${dur} s</small></span>
      <span class="b">${esc(r.server ?? "?")} serving${r.winner ? ` · won by ${esc(r.winner)} <span class="badge recon">tracking</span>` : ""}</span>`;
  }
  function renderList() {
    build();
    const ol = $("#plList", root);
    ol.setAttribute("aria-label", title || "Clips");
    ol.innerHTML = items.length ? items.map((it, i) => `<li${i === idx ? ' aria-current="true"' : ""}><button type="button" data-i="${i}">${itemHTML(it, i)}</button></li>`).join("")
      : `<li class="player-idle" style="border:0">No clips in this selection.</li>`;
    $$("button[data-i]", ol).forEach(b => b.onclick = () => play(+b.dataset.i));
    updateNow();
  }
  function updateNow() {
    $("#plTitle", root).textContent = title || "Clips";
    $("#plPos", root).textContent = items.length ? (idx >= 0 ? `Clip ${idx + 1} of ${items.length}` : `${items.length} clip${items.length === 1 ? "" : "s"} · press a row or Watch`) : "";
    ["#plPrev", "#plNext", "#plReplay"].forEach(s => $(s, root).disabled = !items.length);
    const it = items[idx];
    let d = "";
    if (it && it.kind === "shot") {
      const s = it.s, L = PB.shotLine(s, it.view), o = s.pl.origin;
      const evName = it.view === "interception" ? "the interception" : s.pl.status === "bounced" ? "the bounce" : "the next contact";
      d = `Rally ${rno(s.rid)} · ${cap(PB.ROLE_NOUN[s.pl.role]).replace(/s$/, "")}${s.stroke ? " · " + s.stroke : ""} · hit from ${esc(o ? o.zone : "?")}${o && /proxy/.test(o.evidence || "") ? " (feet proxy)" : ""} → ${esc(L.path.split(" → ")[1])}${L.mph != null ? ` · estimated ${L.mph} mph` : " · speed unknown"}. Clip runs from ${PB.LEAD_S} s before the hit to about ${PB.FOLLOW_S} s after ${evName}.`;
    } else if (it && it.kind === "rally") d = `Whole rally: from 1 s before the serve to 1.5 s after the last shot.`;
    $("#plDesc", root).innerHTML = d;
    const cur = $$("#plList li", root)[idx];
    $$("#plList li", root).forEach(li => li.removeAttribute("aria-current"));
    if (cur) { cur.setAttribute("aria-current", "true"); const ol = $("#plList", root); if (cur.offsetTop < ol.scrollTop || cur.offsetTop > ol.scrollTop + ol.clientHeight - 40) ol.scrollTop = cur.offsetTop - 40; }
    $$(".court .pt").forEach(p => p.classList.toggle("cur", it && it.kind === "shot" && p.dataset.t === String(it.s.t) && +p.dataset.r === it.s.rid));
  }
  function play(i) {
    build();
    const it = items[i]; if (!it) return;
    idx = i;
    if (key.startsWith("sel|")) { ST.clip = String(i + 1); history.replaceState(null, "", PB.stateToSearch(ST)); }
    const w = win(it);
    if (!srcSet) { const src = (R && R.video) || M.video; if (src) { video.src = "../" + src; srcSet = true; } }
    $("#vmsg", root).hidden = !(!srcSet);
    const go = () => { video.currentTime = w.start; stopAt = w.end; const p = video.play(); if (p) p.catch(() => {}); };
    video.readyState >= 1 ? go() : video.addEventListener("loadedmetadata", go, { once: true });
    if (video.readyState < 1) video.load();
    updateNow();
  }
  return {
    mount(slot) { if (slot && build().parentNode !== slot) slot.appendChild(root); $("#plBack", root).hidden = !(slot && slot.id === "plSlot") || innerWidth > 1060; },
    load(list, meta) { build(); const changed = meta.key !== key; items = list; key = meta.key; title = meta.title; if (changed) { idx = meta.idx ?? -1; if (!video.paused) video.pause(); stopAt = null; } renderList(); },
    play, get key() { return key; }, get count() { return items.length; }, refresh: updateNow,
  };
})();

// ------------------------------------------------------------------ overview
function renderOverview() {
  const sec = $("#sec-overview"), shots = PB.shotsFor(R, SEL);
  const obs = PB.observations(R, SEL, { shots, who: whoNoun() });
  const val = PB.venueValidation(M.tournament);
  const sp = PB.speedStats(shots), rallies = new Set(shots.map(s => s.rid)).size;
  const P = R.players[SEL.side], idxC = PB.sidesOf(R).indexOf(SEL.side) + 1;
  // outcome summary: tracked rally wins only where reliable; otherwise the official result
  let outcome;
  if (winnersReliable()) {
    const dec = R.rallies.filter(r => r.winner), won = dec.filter(r => r.winner === SEL.side).length;
    outcome = `<div><div class="l">Rallies won <span class="badge recon">from tracking</span></div><div class="v">${won}<small> of ${dec.length}</small></div><div class="n">${pct(won, dec.length)}% · ${esc(M.check || "")}</div></div>`;
  } else {
    const off = (R.official || M.result);
    outcome = `<div><div class="l">Result <span class="badge official">official</span></div><div class="v" style="font-size:16px">${esc(off || "Not available")}</div><div class="n">${R.type === "doubles" ? "Rally-by-rally winners from tracking are unreliable in doubles (49–78% right), so they are not summarised." : `Tracked rally winners are not summarised here: ${esc(M.check || "the cross-check is weak")}.`}</div></div>`;
  }
  let move = "";
  if (SEL.kind === "side" && P && P.net && P.net.rallies && R.type !== "doubles") move = `<div><div class="l">Reached the net <span class="badge est">estimated</span></div><div class="v">${P.net.reached_pct}%<small> of rallies</small></div><div class="n">${P.net.rallies} rallies · median ${P.net.median_arrival_s ?? "–"} s after the serve</div></div>`;
  else if (SEL.kind === "side" && P && P.running && P.running.rallies) move = `<div><div class="l">Distance run per rally <span class="badge est">estimated</span></div><div class="v">${P.running.dist_per_rally_ft}<small> ft</small></div><div class="n">${R.type === "doubles" ? "team average of both partners" : `${P.running.rallies} rallies`}</div></div>`;
  const byPlayer = SEL.kind === "side" && R.type === "doubles" && R.teams && R.teams[SEL.side] ? (() => {
    const c = {}; let none = 0; shots.forEach(s => s.hitter ? c[s.hitter] = (c[s.hitter] || 0) + 1 : none++);
    return `<p class="note" style="margin:10px 0 0">Shots by player: ${R.teams[SEL.side].map(n => `${esc(n)} <b>${c[n] || 0}</b>`).join(" · ")}${none ? ` · hitter not identified <b>${none}</b>` : ""}. Switch to a player in “Analyse” for individual placement.</p>`;
  })() : "";
  const saved = window.PBBrief ? PBBrief.list(M.id, SEL.key) : [];
  sec.innerHTML = `<div class="head"><h2 id="h-overview">Key observations</h2><p>${esc(whoNoun())} · descriptive tendencies from the tracking in this match - not tactical advice.</p></div>
    ${obs.length ? `<div class="obs-grid">${obs.map((o, i) => obsCard(o, i, val)).join("")}</div>`
      : `<div class="state-empty obs-none"><b>No observation is well enough supported to headline.</b> ${shots.length} analysed shots for ${esc(whoNoun())}; an observation needs at least ${PB.OBS_MIN.landingRole}–${PB.OBS_MIN.landingAll} recorded locations and one clearly most common zone. Explore the numbers directly in <button type="button" class="linkbtn" data-go="placement">Placement &amp; speed</button>.</div>`}
    <p class="note" style="margin:8px 0 0">${obs.length < 3 && obs.length ? `Only ${obs.length} observation${obs.length > 1 ? "s are" : " is"} well supported for this selection, so only ${obs.length > 1 ? "those are" : "that is"} shown. ` : ""}Observations use fixed minimum samples and never rest on shot-type labels, reconstructed winners or spin.${saved.length ? ` <a href="${$("#briefBtn").href}">${saved.length} saved to the coaching brief on this device →</a>` : ""}</p>
    <h3 style="margin:22px 0 8px;font-size:15px;color:var(--paddle)">At a glance</h3>
    <div class="sumstrip">
      <div><div class="l">Analysed</div><div class="v">${shots.length}<small> shots</small></div><div class="n">in ${rallies} of ${R.rallies.length} rallies found · ${esc(M.check || "")}</div></div>
      ${outcome}
      <div><div class="l">Estimated speed <span class="badge est">estimated</span></div><div class="v">${sp.median != null ? `${sp.median}<small> mph median</small>` : "Unknown"}</div><div class="n">usable for ${sp.usable} of ${shots.length} shots · launch speed off the paddle, not radar-checked</div></div>
      ${move}
    </div>${byPlayer}
    ${DB ? partnershipGlance() : ""}
    ${R.rotations ? rotationsBlock() : ""}
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:18px"><button type="button" class="btn primary" data-go="placement">Explore placement &amp; speed →</button><button type="button" class="btn" data-go="clips">Watch rallies</button><button type="button" class="btn ghost" id="ovDQ">Data quality</button></div>`;
  $$("[data-go]", sec).forEach(b => b.onclick = () => go({ sec: b.dataset.go }));
  $("#ovDQ", sec).onclick = openDQ;
  $$(".obs [data-watch]", sec).forEach(b => b.onclick = () => {
    const o = obs[+b.dataset.watch];
    go(Object.assign({ sec: "placement", role: "all", stroke: "", incoming: "", status: "", ev: "", zone: "", mode: "zones" }, o.link));
    Player.play(0);
    if (innerWidth <= 1060) $("#plSlot")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $$(".obs [data-save]", sec).forEach(b => b.onclick = () => {
    const o = obs[+b.dataset.save], id = `${M.id}|${SEL.key}|${o.id}`;
    if (PBBrief.has(id)) PBBrief.remove(id);
    else PBBrief.add({ id, match: M.id, who: SEL.key, auto: true, headline: o.headline, measure: o.measure, scope: o.scope, quality: o.quality,
      link: `hub.html${PB.stateToSearch(Object.assign({ m: M.id, p: SEL.key, role: "all", view: "landing", mode: "zones" }, o.link))}`, clips: obsClips(o) });
    renderOverview();
  });
}
function obsClips(o) {
  const f = Object.assign({ role: "all" }, o.link), agg = PB.aggregate(PB.applyFilters(PB.shotsFor(R, SEL), f, f.view), f.view);
  return PB.selectShots(agg, f.zone || "").slice(0, 8).map(s => ({ rally: s.rid, t: Math.round(PB.clipWindow(R, s, f.view).hit) }));
}
function obsCard(o, i, val) {
  const saved = window.PBBrief && PBBrief.has(`${M.id}|${SEL.key}|${o.id}`);
  const vq = o.kind === "landing" && val ? [val.short] : [];
  return `<article class="obs" aria-labelledby="obs${i}">
    <div class="badges">${o.evidence === "human" ? '<span class="badge human">Human-reviewed</span>' : o.evidence === "proxy" ? '<span class="badge auto">Automatically detected</span><span class="badge proxy">Includes position proxies</span>' : '<span class="badge auto">Automatically detected</span>'}${o.limited ? '<span class="badge limited">Limited sample</span>' : ""}</div>
    <h3 id="obs${i}">${esc(o.headline)}</h3>
    <div class="measure">${esc(o.measure)}</div>
    <div class="scope">${esc(o.scope)}</div>
    <ul class="q">${[...o.quality, ...vq].map(q => `<li>${esc(q)}</li>`).join("")}</ul>
    <div class="actions"><button type="button" class="btn primary sm" data-watch="${i}">Watch supporting shots →</button>
      <button type="button" class="btn sm" data-save="${i}" aria-pressed="${!!saved}">${saved ? "✓ Saved to brief" : "Save to brief"}</button></div>
  </article>`;
}
function rotationsBlock() {
  const [t0, t1] = R.overlay_teams || PB.sidesOf(R);
  return `<details class="more" style="margin-top:16px"><summary>Matchups by rotation <span class="s">from the broadcast score overlay · ${R.rotations.reduce((x, r) => x + r.points[0] + r.points[1], 0)} points</span></summary><div class="body tablewrap">
    <p class="note" style="margin-top:0">Players rotate every 5 points in a fixed order, so the official score says who was on court for each point (including points the tracking missed).</p>
    <table class="t"><tr><th>Turn</th><th>${esc(t0)}</th><th>${esc(t1)}</th><th class="r">Points</th></tr>
    ${R.rotations.map(r => `<tr><td>${esc(r.slot)}</td><td>${esc(r.players[0])}</td><td>${esc(r.players[1])}</td><td class="r">${r.points[0]}–${r.points[1]}</td></tr>`).join("")}</table></div></details>`;
}

function partnershipGlance() {
  const T0 = DB.teams[SEL.side]; if (!T0) return "";
  const me = SEL.kind === "player" ? DB.players[SEL.key] : null;
  const share = Object.entries(DB.players).filter(([, p]) => p.team === SEL.side).map(([n, p]) => `${esc(n)} ${p.shots.team_rally_share_pct ?? "–"}%`).join(" · ");
  return `<h3 style="margin:22px 0 8px;font-size:15px;color:var(--paddle)">${me ? `${esc(SEL.key)} in the partnership` : "Partnership at a glance"} <span class="badge est">estimated</span></h3>
    <div class="sumstrip">
      ${me ? `<div><div class="l">Up (within ${DB.up_ft} ft of the net)</div><div class="v">${me.position.up_pct ?? "–"}%<small> of rally time</small></div><div class="n">avg ${me.position.avg_depth_ft ?? "–"} ft from the net</div></div>
        <div><div class="l">Played on the left</div><div class="v">${me.position.left_pct ?? "–"}%<small> of rally time</small></div><div class="n">their own left half</div></div>
        <div><div class="l">Share of the team's rally shots</div><div class="v">${me.shots.team_rally_share_pct ?? "–"}%</div><div class="n">${me.shots.rally_shots} of the team's attributed rally shots</div></div>`
      : `<div><div class="l">Both partners up</div><div class="v">${T0.formation_time_pct.both_up ?? "–"}%<small> of rally time</small></div><div class="n">both back ${T0.formation_time_pct.both_back ?? "–"}% · ${T0.pair_rallies} rallies measured</div></div>
        <div><div class="l">Distance between partners</div><div class="v">${T0.spacing.median_ft ?? "–"}<small> ft median</small></div><div class="n">${T0.spacing.lateral_ft ?? "–"} ft sideways</div></div>
        <div><div class="l">Rally shots by player</div><div class="v" style="font-size:16px">${share || "–"}</div><div class="n">of ${T0.shots.rally} attributed rally shots</div></div>`}
    </div><p class="note" style="margin:8px 0 0"><button type="button" class="linkbtn" data-go="partnership">Open Partnership →</button> Each player and the pair together, compared with the opponents.</p>`;
}

// doubles matches processed without a report: status only
async function renderStatus() {
  const sec = $("#sec-overview");
  let ans = null;
  try { ans = await getJSON(`/api/labels/${M.queue}.shots.json`); } catch (e) { /* served by serve.py only */ }
  const s = A ? A.summary : null;
  sec.innerHTML = `<div class="head"><h2 id="h-overview">Overview</h2><p>${reportError ? "The match report could not be loaded." : "This match is processed, but no placement report has been built for it yet."}</p></div>
    <div class="cols"><div class="panel"><h3>Status</h3>
      ${s ? `<div class="mtiles" style="margin-top:8px"><div class="mtile"><div class="l">Rallies found</div><div class="v">${s.rallies}</div><div class="n">${esc(M.check || "")}</div></div>
        <div class="mtile"><div class="l">Shots detected</div><div class="v">${s.shots}</div><div class="n">hits found 86–94% (validated)</div></div>
        <div class="mtile"><div class="l">Shots reviewed by you</div><div class="v">${ans && ans.answers ? Object.keys(ans.answers).length : 0}</div><div class="n">${M.queue ? `<a href="review.html?video=${encodeURIComponent(M.queue)}">Open the review queue →</a>` : ""}</div></div></div>`
        : `<div class="state-empty">Still being processed - no analysis file yet.</div>`}
      <p class="note" style="margin-bottom:0">Placement & speed, Movement and Clips need a match report (per-shot placement records). They are disabled for this match rather than shown empty.</p></div>
      <div class="panel"><h3>Match video</h3>${M.video ? `<div class="player"><div class="vwrap"><video controls preload="none" playsinline poster="${esc(M.thumb || "")}" src="../${esc(M.video)}"></video></div></div>` : `<div class="state-empty">No video linked.</div>`}</div></div>`;
}

// ------------------------------------------------------------------ placement & speed
function renderPlacement() {
  const sec = $("#sec-placement"), { all, shots, agg, list } = selectionModel(), V = VIEWS[ST.view];
  const roleCounts = Object.fromEntries(ROLES.map(([k]) => [k, PB.applyFilters(all, Object.assign(curFilters(), { role: k }), ST.view).length]));
  const narrow = matchMedia("(max-width: 520px)").matches;
  const moreN = ["stroke", "incoming", "status", "ev"].filter(k => ST[k]).length;
  const chip = (k, label, na) => `<span class="xchip${na ? " na" : ""}">${esc(label)}${na ? " · not used in this view" : ""}<button type="button" data-clear="${k}" aria-label="Remove filter: ${esc(label)}">×</button></span>`;
  const chips = [ST.role !== "all" && chip("role", ROLES.find(r => r[0] === ST.role)[1]),
    ST.stroke && chip("stroke", cap(ST.stroke)), ST.incoming && chip("incoming", "Incoming: " + PB.INCOMING.find(x => x[0] === ST.incoming)[1]),
    ST.status && chip("status", "Outgoing: " + PB.STATUSES.find(x => x[0] === ST.status)[1]), ST.ev && chip("ev", "Evidence: " + PB.EVIDENCE[ST.view].find(x => x[0] === ST.ev)[1]),
    ST.zone && chip("zone", "Zone: " + (ST.zone === "outside" ? "outside the court" : ST.zone === "otherhalf" ? "other half" : zoneLabel(ST.zone)))].filter(Boolean);
  const sel = (k, label, opts, hint) => `<label>${label}<select data-f="${k}"><option value="">Any</option>${opts.map(([v, l]) => `<option value="${esc(v)}"${ST[k] === v ? " selected" : ""}>${esc(l)}</option>`).join("")}</select>${hint ? `<span class="hint">${hint}</span>` : ""}</label>`;
  const sp = PB.speedStats(list), den = agg.denominator;
  const val = PB.venueValidation(M.tournament);
  const human = list.filter(s => PB.evidenceOf(s, ST.view) === "human").length, proxy = list.filter(s => PB.evidenceOf(s, ST.view) === "proxy").length;
  const share = ST.zone ? pct(list.length, den) : null;
  const briefId = `${M.id}|${SEL.key}|${selKey()}`;
  sec.innerHTML = `<div class="head"><h2 id="h-placement">Placement &amp; speed</h2><p>${esc(whoNoun())} · every location is seen from behind the hitter, looking over the net.</p></div>
    <div class="filterbar">
      <span class="lbl" id="sitLbl">Situation</span>
      <div class="seg" role="group" aria-labelledby="sitLbl">${ROLES.map(([k, l]) => `<button type="button" data-role="${k}" aria-pressed="${ST.role === k}">${l} <span class="n">${roleCounts[k]}</span></button>`).join("")}</div>
      <details class="morebox" id="moreBox"><summary class="btn sm" aria-label="More filters${moreN ? `, ${moreN} active` : ""}">More filters${moreN ? ` (${moreN})` : ""} ▾</summary>
        <div class="morepanel">
          ${sel("stroke", "Stroke", PB.STROKES)}
          ${sel("incoming", "Incoming ball", PB.INCOMING)}
          ${ST.view === "origin" ? sel("status", "Outgoing flight", PB.STATUSES, "Only in the Hit from view - Landings and Interceptions already are one outcome each.") : `<p class="hint" style="margin:0">Outgoing-flight filter: available in the <b>Hit from</b> view.</p>`}
          ${sel("ev", `Evidence of the ${ST.view === "origin" ? "contact point" : ST.view === "landing" ? "landing" : "interception"}`, PB.EVIDENCE[ST.view], "Options differ per map view.")}
        </div></details>
      <div class="activechips" aria-live="polite">${chips.join("")}${chips.length ? `<button type="button" class="btn ghost sm" id="resetF">Reset filters</button>` : ""}</div>
      <span class="matchcount"><b>${shots.length}</b> matching shot${shots.length === 1 ? "" : "s"}</span>
    </div>
    ${FLASH.map(f => `<div class="flash" role="status">${esc(f)}</div>`).join("")}
    <div class="pl-layout">
      <div class="panel" id="courtPanel">
        <div class="courtbar">
          <div class="seg" role="group" aria-label="Map view">${Object.entries(VIEWS).map(([k, v]) => `<button type="button" data-view="${k}" aria-pressed="${ST.view === k}">${v.label}</button>`).join("")}</div>
          <div class="seg sm" role="group" aria-label="Map display">${[["zones", "Zones"], ["shots", "Individual shots"]].map(([k, l]) => `<button type="button" data-mode="${k}" aria-pressed="${ST.mode === k}">${l}</button>`).join("")}</div>
        </div>
        <p class="orient"><b>${V.label}:</b> ${esc(V.def)} ${ST.view === "origin" ? "Hitter's own half: net at the top, their baseline at the bottom." : "Opponent's half: net at the bottom, the opponent's baseline at the top."} Left = the hitter's left.</p>
        ${den ? courtSVG(agg, narrow) : `<div class="state-empty">No ${V.noun} for this selection.${shots.length ? ` ${shots.length} matching shots, but none ${ST.view === "landing" ? "has an observed first bounce" : ST.view === "interception" ? "was intercepted with a recorded position" : "has a contact point"}.` : " Try another situation or reset the filters."}</div>`}
        ${den ? legendHTML(den) : ""}
        ${notInZoneHTML(agg)}
        ${statusLine(agg)}
      </div>
      <div>
        <div class="panel selpanel" aria-live="polite">
          <div class="slabel">${esc(selectionLabel())}</div>
          <div class="big"><span class="num1">${ST.zone ? list.length : den}</span><span class="of">${ST.zone ? `of ${den} ${V.noun}${share != null ? ` (${share}%)` : ""}` : `${V.noun} · of ${agg.population} ${V.popNoun}${agg.missing.length ? `; ${agg.missing.length} without a recorded position` : ""}`}</span></div>
          <div class="speedrow">${list.length ? speedText(sp, ST.view === "landing" ? "landings" : ST.view === "interception" ? "interceptions" : "shots") : ""}</div>
          <div class="qline" style="margin-bottom:10px">${list.length ? `${human ? `<span>${evBadge("human", "landing")} ${human}</span>` : ""}<span>${evBadge(ST.view === "landing" ? "auto2" : "ball", ST.view)} ${list.length - human - proxy}</span>${proxy ? `<span>${evBadge("proxy", ST.view)} ${proxy}</span>` : ""}` : ""}${ST.view === "landing" && val ? `<span class="note">${esc(val.short)}</span>` : ST.view === "interception" ? `<span class="note">Interception positions are not validated against hand tags; many sit on a zone boundary.</span>` : `<span class="note">Contact points: ball at contact (3D fit) or the hitter's feet as a proxy; not validated against hand tags.</span>`}</div>
          <div class="selactions">
            <button type="button" class="btn primary" id="watchBtn"${list.length ? "" : " disabled"}>Watch ${list.length} shot${list.length === 1 ? "" : "s"} →</button>
            ${ST.zone ? `<button type="button" class="btn" id="clearZone">Clear zone</button>` : `<span class="note">Click a zone (or a row label) on the court to narrow down.</span>`}
            <button type="button" class="btn ghost" id="saveSel" aria-pressed="${PBBrief.has(briefId)}"${list.length ? "" : " disabled"}>${PBBrief.has(briefId) ? "✓ Saved to brief" : "Save to brief"}</button>
          </div>
        </div>
        <div class="panel" id="plSlot" style="margin-top:16px" aria-label="Video and matching shots"></div>
      </div>
    </div>
    <div style="margin-top:16px">${speedDetails(all)}${pairsDetails(shots)}
      <details class="more"><summary>Definitions <span class="s">zones, views, denominators</span></summary><div class="body note">
        Zones (from the net, on each half): kitchen 0–7 ft, midcourt 7–14.5 ft, deep 14.5–22 ft; left / centre / right thirds of the 20 ft width, named from the hitter's side. A location within one position-uncertainty of a zone line is reported as <b>on a zone boundary</b> and is never forced into a zone. Percentages on the map are shares of all ${esc(V.noun)} shown (boundary and outside ones included), so the zones plus the “not in a zone” groups add up to 100%. Estimated speed = launch speed off the paddle from the physics fit of the outgoing flight, whole mph, not radar-checked. <button type="button" class="linkbtn" id="defDQ">Open data quality</button></div></details>
    </div>`;
  // events
  $$("[data-role]", sec).forEach(b => b.onclick = () => go({ role: b.dataset.role, zone: "" }));
  $$("[data-view]", sec).forEach(b => b.onclick = () => { if (b.dataset.view === ST.view) return; const r = PB.switchView(ST, b.dataset.view); FLASH = r.notes; go(r.state, { keepFlash: true }); });
  $$("[data-mode]", sec).forEach(b => b.onclick = () => go({ mode: b.dataset.mode }));
  $$("select[data-f]", sec).forEach(s => s.onchange = () => go({ [s.dataset.f]: s.value, zone: "" }));
  $$("[data-clear]", sec).forEach(b => b.onclick = () => go({ [b.dataset.clear]: b.dataset.clear === "role" ? "all" : "" }));
  $("#resetF", sec) && ($("#resetF", sec).onclick = () => go({ role: "all", stroke: "", incoming: "", status: "", ev: "", zone: "" }));
  $("#clearZone", sec) && ($("#clearZone", sec).onclick = () => go({ zone: "" }));
  $("#defDQ", sec).onclick = openDQ;
  $("#moreBox", sec).addEventListener("keydown", e => { if (e.key === "Escape") { e.currentTarget.open = false; $("summary", e.currentTarget).focus(); } });
  bindCourt(sec);
  $$("[data-nz]", sec).forEach(b => b.onclick = () => go({ zone: ST.zone === b.dataset.nz ? "" : b.dataset.nz }));
  $("#watchBtn", sec).onclick = () => { Player.play(0); if (innerWidth <= 1060) $("#plSlot").scrollIntoView({ behavior: "smooth", block: "start" }); };
  $("#saveSel", sec).onclick = () => {
    if (PBBrief.has(briefId)) PBBrief.remove(briefId);
    else PBBrief.add({ id: briefId, match: M.id, who: SEL.key, auto: false, headline: selectionLabel(),
      measure: ST.zone ? `${list.length} of ${den} ${V.noun} (${share}%)` : `${den} ${V.noun} of ${agg.population} ${V.popNoun}`,
      scope: `${whoNoun()} · ${M.title}`, quality: [sp.usable ? `Estimated speed median ${sp.median} mph (usable ${sp.usable} of ${sp.n})` : "No usable speed estimate", ...(ST.view === "landing" && val ? [val.short] : [])],
      link: `hub.html${PB.stateToSearch(Object.assign({}, ST, { sec: "placement", clip: "" }))}`, clips: list.slice(0, 8).map(s => ({ rally: s.rid, t: Math.round(PB.clipWindow(R, s, ST.view).hit) })) });
    renderPlacement();
  };
  $$("[data-pair]", sec).forEach(b => b.onclick = () => { const [k, key] = b.dataset.pair.split("::"); const arr = pairSets[k][key]; watchCustom($("#plSlot"), shotItems(arr, k === "landing" ? "landing" : "interception"), `${key} · ${ROLE_NOUN[ST.role]}`, `pair|${k}|${key}|${selKey()}`); });
  $$("[data-spd]", sec).forEach(b => b.onclick = () => { const arr = spdSets[b.dataset.spd]; watchCustom($("#plSlot"), shotItems(arr, ST.view), `${b.dataset.spd} · fastest first`, `spd|${b.dataset.spd}|${selKey()}`); });
  // player
  Player.mount($("#plSlot"));
  const k = selKey();
  if (Player.key !== k) Player.load(shotItems(list, ST.view), { key: k, title: selectionLabel(), idx: ST.clip ? Math.min(list.length, +ST.clip) - 1 : -1 });
  else Player.refresh();
}
function binOf(share) { return share <= 0 ? 0 : share < 10 ? 1 : share < 20 ? 2 : share < 30 ? 3 : 4; }
function legendHTML(den) {
  const V = VIEWS[ST.view];
  if (ST.mode === "zones") return `<div class="legend" aria-label="Colour legend"><span><span class="sw" style="background:var(--zs-0)"></span>0%</span><span><span class="sw" style="background:var(--zs-1)"></span>under 10%</span><span><span class="sw" style="background:var(--zs-2)"></span>10–19%</span><span><span class="sw" style="background:var(--zs-3)"></span>20–29%</span><span><span class="sw" style="background:var(--zs-4)"></span>30% or more</span>
    <span>Share of the <b>${den}</b> ${esc(V.noun)}; count under each %. Same scale for every player.</span></div>`;
  const m = (shape) => `<svg width="16" height="16" viewBox="-8 -8 16 16" aria-hidden="true">${markerShape(shape, 0, 0, false)}</svg>`;
  const items = ST.view === "landing" ? [["human", "Human-reviewed bounce"], ["auto", "Automatically detected bounce"]] : [["auto", "Ball at contact (3D fit)"], ["proxy", ST.view === "origin" ? "Hitter's feet (position proxy)" : "Opponent's feet (position proxy)"]];
  return `<div class="legend" aria-label="Marker legend">${[...items, ["boundary", "On a zone boundary"], ["outside", "Outside the court / other half (edge of map)"]].map(([s, l]) => `<span>${m(s)}${l}</span>`).join("")}<span>Click a marker to play it. Every shot is also in the playlist.</span></div>`;
}
function markerShape(kind, x, y, cur) {
  const fill = cur ? "var(--ball)" : kind === "human" ? "var(--court-green)" : "#fff";
  const stroke = kind === "proxy" ? "#8a5a00" : kind === "boundary" ? "var(--paddle)" : kind === "outside" ? "#b3261e" : "var(--court-green)";
  if (kind === "proxy") return `<rect class="ring" x="${x - 5}" y="${y - 5}" width="10" height="10" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
  if (kind === "boundary") return `<path class="ring" d="M${x},${y - 6.5} L${x + 6.5},${y} L${x},${y + 6.5} L${x - 6.5},${y} Z" fill="${fill}" stroke="${stroke}" stroke-width="1.8"/>`;
  if (kind === "outside") return `<g class="ring" stroke="${stroke}" stroke-width="2.6" stroke-linecap="round"><path d="M${x - 5},${y - 5} L${x + 5},${y + 5} M${x + 5},${y - 5} L${x - 5},${y + 5}"/></g>${cur ? `<circle cx="${x}" cy="${y}" r="7" fill="none" stroke="var(--paddle)" stroke-width="2"/>` : ""}`;
  return `<circle class="ring" cx="${x}" cy="${y}" r="${cur ? 7 : 5.5}" fill="${fill}" stroke="${kind === "human" ? "#fff" : stroke}" stroke-width="${kind === "human" ? 1.5 : 2}"/>`;
}
// Court drawn from behind the hitter. Opponent-half views: far baseline at the top, net at the bottom (with a strip
// of the hitter's own side below it). Hit from: net at the top (a strip of the opponent's side above it), the
// hitter's baseline and the area behind it at the bottom. Left of the drawing = the hitter's left in every view.
function courtSVG(agg, narrow) {
  const view = ST.view, opp = view !== "origin", mode = ST.mode, den = agg.denominator;
  const s = 16, mx = narrow ? 0.9 : 2.4, gut = narrow ? 76 : 82, top = 26, strip = 2;
  const dMax = opp ? 22 + mx : 29.5;
  const W = gut + (20 + 2 * mx) * s + 8;
  const Y = d => opp ? top + (dMax - d) * s : top + (strip + d) * s;
  const H = (opp ? Y(-strip) : Y(dMax)) + 34;
  const X = u => gut + (mx + u) * s;
  const yA = (d0, d1) => [Math.min(Y(d0), Y(d1)), Math.abs(Y(d1) - Y(d0))];
  let g = `<rect x="${X(-mx)}" y="${Y(opp ? dMax : -strip)}" width="${(20 + 2 * mx) * s}" height="${Math.abs(Y(opp ? -strip : dMax) - Y(opp ? dMax : -strip))}" rx="6" fill="#e3eae4"/>`;
  // the other side of the net (context only)
  const [sy, sh] = yA(0, -strip);
  g += `<rect x="${X(-mx)}" y="${sy}" width="${(20 + 2 * mx) * s}" height="${sh}" fill="#f3f5f2"/><text x="${X(10)}" y="${sy + sh / 2 + 4}" text-anchor="middle" font-size="11.5" fill="var(--text-muted)">${opp ? "hitter's side" : "opponent's side"}</text>`;
  // zone cells
  const rows = PB.ROWS[view];
  for (const d of rows) {
    const [b0, b1] = PB.BANDS[d], [y0, h] = yA(b0, Math.min(b1, d === "behind" ? dMax : b1));
    for (const [wi, w] of PB.WIDTHS.entries()) {
      const z = `${d} ${w}`, list = agg.cells[z] || [], n = list.length, sh2 = den ? 100 * n / den : 0, bin = mode === "zones" ? binOf(sh2) : 0;
      const x0 = X(wi * 20 / 3), cw = 20 / 3 * s, sel = ST.zone === z;
      const dark = bin >= 3, tc = dark ? "#fff" : "var(--paddle)", tc2 = dark ? "#e8f1ea" : "var(--text-secondary)";
      g += `<g class="zc${sel ? " sel" : ""}" data-z="${z}" role="button" tabindex="0" aria-pressed="${sel}" aria-label="${esc(zoneLabel(z))}: ${n} of ${den} ${VIEWS[view].noun}${den ? ` (${Math.round(sh2)}%)` : ""}">
        <rect class="zr" x="${x0}" y="${y0}" width="${cw}" height="${h}" fill="var(--zs-${bin})" ${d === "behind" ? 'fill-opacity="0.75"' : ""} stroke="#fff" stroke-width="2"/>
        ${mode === "zones" ? (n ? `<text x="${x0 + cw / 2}" y="${y0 + h / 2 + 3}" text-anchor="middle" font-size="${narrow ? 23 : 19}" font-weight="750" fill="${tc}">${Math.round(sh2)}%</text><text x="${x0 + cw / 2}" y="${y0 + h / 2 + (narrow ? 23 : 20)}" text-anchor="middle" font-size="${narrow ? 15 : 12.5}" fill="${tc2}">${n} shot${n === 1 ? "" : "s"}</text>`
          : `<text x="${x0 + cw / 2}" y="${y0 + h / 2 + 5}" text-anchor="middle" font-size="13" fill="var(--text-muted)">0</text>`) : ""}
        <rect class="zsel" x="${x0 + 3}" y="${y0 + 3}" width="${cw - 6}" height="${h - 6}" fill="none" stroke="var(--ball)" stroke-width="5" pointer-events="none"/>
      </g>`;
    }
    // row label (selects the whole row)
    const selB = ST.zone === `band:${d}`, [ry, rh] = yA(b0, Math.min(b1, d === "behind" ? dMax : b1));
    const cnt = PB.WIDTHS.reduce((x, w) => x + (agg.cells[`${d} ${w}`] || []).length, 0);
    g += `<g class="rowlbl" data-z="band:${d}" role="button" tabindex="0" aria-pressed="${selB}" aria-label="Whole ${d} row: ${cnt} of ${den}${den ? ` (${pct(cnt, den)}%)` : ""}">
      <rect x="2" y="${ry + 2}" width="${gut - 10}" height="${rh - 4}" rx="6" fill="${selB ? "var(--lime-bg)" : "transparent"}" stroke="${selB ? "var(--paddle)" : "transparent"}" stroke-width="1.5"/>
      <text x="${gut - 10}" y="${ry + rh / 2 - 1}" text-anchor="end" font-size="${narrow ? 15 : 13}" font-weight="650" fill="var(--paddle)">${d === "behind" ? "Behind" : cap(d)}</text>
      <text x="${gut - 10}" y="${ry + rh / 2 + 15}" text-anchor="end" font-size="${narrow ? 13 : 11.5}" fill="var(--text-muted)">${d === "behind" ? "baseline" : d === "kitchen" ? "0–7 ft" : d === "midcourt" ? "7–14.5 ft" : "14.5–22 ft"}</text></g>`;
    if (selB) { const [by, bh] = yA(b0, Math.min(b1, d === "behind" ? dMax : b1)); g += `<rect x="${X(0) + 3}" y="${by + 3}" width="${20 * s - 6}" height="${bh - 6}" fill="none" stroke="var(--ball)" stroke-width="5" pointer-events="none"/><rect x="${X(0)}" y="${by}" width="${20 * s}" height="${bh}" fill="none" stroke="var(--paddle)" stroke-width="2.5" pointer-events="none"/>`; }
  }
  // real court lines over the zone grid
  const L = (u0, d0, u1, d1, w, extra = "") => `<line x1="${X(u0)}" y1="${Y(d0)}" x2="${X(u1)}" y2="${Y(d1)}" stroke="#1f4d32" stroke-width="${w}" ${extra} pointer-events="none"/>`;
  // centre service line only as stubs at the kitchen line and baseline, so it never crosses the zone numbers
  g += L(0, 0, 0, 22, 2.5) + L(20, 0, 20, 22, 2.5) + L(0, 22, 20, 22, 2.5) + L(0, 7, 20, 7, 2.5) + L(10, 7, 10, 8.4, 2) + L(10, 20.6, 10, 22, 2);
  // net
  g += `<line x1="${X(-mx) + 2}" y1="${Y(0)}" x2="${X(20 + mx) - 2}" y2="${Y(0)}" stroke="var(--paddle)" stroke-width="5" stroke-linecap="round" pointer-events="none"/>`;
  g += `<rect x="${X(20 + mx) - 42}" y="${Y(0) - 10}" width="38" height="20" rx="4" fill="var(--paddle)"/><text x="${X(20 + mx) - 23}" y="${Y(0) + 4.5}" text-anchor="middle" font-size="11.5" font-weight="700" fill="#fff" letter-spacing=".06em">NET</text>`;
  // labels
  g += opp ? `<text x="${X(10)}" y="${top - 9}" text-anchor="middle" font-size="12.5" fill="var(--text-secondary)">Opponent's baseline ↑ (far end)</text>`
    : `<text x="${X(20 + mx) - 4}" y="${Y(22) - 6}" text-anchor="end" font-size="11.5" fill="var(--text-secondary)">hitter's baseline</text><text x="${X(10)}" y="${top - 9}" text-anchor="middle" font-size="12.5" fill="var(--text-secondary)">Looking toward the net ↑</text>`;
  const by = H - 12;
  g += `<text x="${X(0)}" y="${by}" font-size="12.5" font-weight="600" fill="var(--paddle)">◀ Hitter's left</text><text x="${X(20)}" y="${by}" text-anchor="end" font-size="12.5" font-weight="600" fill="var(--paddle)">Hitter's right ▶</text>`;
  // individual shots
  if (mode === "shots") {
    const inSel = new Set(PB.selectShots(agg, ST.zone));
    for (const sh of agg.recorded) {
      const l = PB.locOf(sh, view), ev = PB.evidenceOf(sh, view);
      let u = l.u, d = l.d, kind = l.half !== VIEWS[view].half ? "outside" : l.zone === "boundary" ? "boundary" : !PB.isCell(l.zone, view) ? "outside" : ev === "proxy" ? "proxy" : ev === "human" ? "human" : "auto";
      if (l.half !== VIEWS[view].half) d = -Math.min(1.6, d);
      const cu = Math.max(-mx + 0.35, Math.min(20 + mx - 0.35, u)), cd = Math.max(-strip + 0.3, Math.min(dMax - 0.3, d));
      const dim = ST.zone && !inSel.has(sh);
      const w = PB.clipWindow(R, sh, view);
      g += `<g class="pt" data-r="${sh.rid}" data-t="${sh.t}" tabindex="-1" opacity="${dim ? 0.22 : 1}"><title>Rally ${rno(sh.rid)} · ${fmtTime(w.hit)} · ${esc(l.zone === "boundary" ? "boundary: " + (l.candidates || []).join(" / ") : l.zone)}${ev ? " · " + PB.EV_BADGE[ev][0] : ""}</title>${markerShape(kind, X(cu), Y(cd), false)}</g>`;
    }
  }
  return `<svg class="court" viewBox="0 0 ${W} ${H}" role="group" aria-label="${esc(VIEWS[view].label)} map, ${den} ${esc(VIEWS[view].noun)}">${g}</svg>`;
}
function bindCourt(sec) {
  const pick = z => go({ zone: ST.zone === z ? "" : z });
  $$(".court .zc, .court .rowlbl", sec).forEach(e => {
    e.addEventListener("click", () => pick(e.dataset.z));
    e.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(e.dataset.z); setTimeout(() => $(`.court [data-z="${CSS.escape(e.dataset.z)}"]`)?.focus(), 0); } });
  });
  $$(".court .pt", sec).forEach(p => p.addEventListener("click", ev => {
    ev.stopPropagation();
    const { list } = selectionModel(), i = list.findIndex(s => s.rid === +p.dataset.r && String(s.t) === p.dataset.t);
    if (i >= 0) Player.play(i);
    else { go({ zone: "" }); const j = selectionModel().list.findIndex(s => s.rid === +p.dataset.r && String(s.t) === p.dataset.t); if (j >= 0) Player.play(j); }
  }));
}
function notInZoneHTML(agg) {
  const V = VIEWS[ST.view], b = [];
  const btn = (k, n, label) => n ? `<button type="button" data-nz="${k}" aria-pressed="${ST.zone === k}"><b>${n}</b> ${label}</button>` : "";
  b.push(btn("boundary", agg.boundary.length, "on a zone boundary"));
  b.push(btn("outside", agg.outside.length, ST.view === "origin" ? "wide of the sidelines" : "estimated outside the court"));
  b.push(btn("otherhalf", agg.otherHalf.length, ST.view === "origin" ? "hit from the opponent's half (likely tracking error)" : "on the hitter's own half (likely tracking error)"));
  const shown = b.filter(Boolean);
  const miss = agg.missing.length ? `<span class="note" style="align-self:center">${agg.missing.length} ${V.popNoun} without a recorded position (unknown, not in the ${agg.denominator})</span>` : "";
  if (!shown.length && !miss) return agg.denominator ? `<p class="note" style="margin:10px 0 0">All ${agg.denominator} ${esc(V.noun)} fall clearly inside one zone.</p>` : "";
  return `<div class="nz" aria-label="Not in a zone">${shown.length ? `<span class="note" style="align-self:center">Not in a zone:</span>${shown.join("")}` : ""}${miss}</div>`;
}
function statusLine(agg) {
  if (!agg.matching) return "";
  const st = agg.status, parts = [["bounced", "bounced first"], ["intercepted", "intercepted"], ["net", "ended at the net"], ["unresolved", "unresolved"]].filter(([k]) => st[k]).map(([k, l]) => `<b>${st[k]}</b> ${l}`);
  const pop = ST.view === "landing" ? "This map shows only the shots that bounced first; interceptions are a separate view and never count as landings." : ST.view === "interception" ? "This map shows only intercepted shots; nothing here is a landing." : "This map shows every matching shot's contact point, whatever happened next.";
  return `<p class="statusline">Outgoing flight of the ${agg.matching} matching shots: ${parts.join(" · ")}. ${pop}</p>`;
}
let spdSets = {}, pairSets = { landing: {}, interception: {} };
function speedDetails(all) {
  const base = PB.applyFilters(all, Object.assign(curFilters(), { role: "all", status: "", ev: "" }), "origin");
  spdSets = {};
  const row = (label, arr) => { const sp = PB.speedStats(arr); if (!arr.length) return ""; spdSets[label] = arr.filter(s => s.pl.speed && s.pl.speed.mph != null).sort((a, b) => b.pl.speed.mph - a.pl.speed.mph);
    return `<tr><td>${esc(label)}${sp.usable && sp.usable < 10 ? ' <span class="badge limited">limited sample</span>' : ""}</td><td class="r">${arr.length}</td><td class="r">${sp.usable}</td><td class="r"><b>${sp.median ?? "–"}</b>${sp.median != null ? " mph" : ""}</td><td class="r">${sp.q1 != null ? `${sp.q1}–${sp.q3}` : "–"}</td><td>${sp.usable ? `<button type="button" class="btn sm" data-spd="${esc(label)}">Watch</button>` : ""}</td></tr>`; };
  const rows = [["By situation", [["Serve", "serve"], ["Return", "return"], ["Third shot", "third"], ["Later rally shots", "later"]].map(([l, k]) => row(l, base.filter(s => s.pl.role === k)))],
    ["By where it was hit from", ["kitchen", "midcourt", "deep", "behind"].map(d => row(`Hit from ${d === "behind" ? "behind the baseline" : d}`, base.filter(s => s.pl.origin && s.pl.origin.half === "own" && s.pl.origin.zone.startsWith(d + " "))))],
    ["By outgoing flight", [["Bounced first", "bounced"], ["Intercepted", "intercepted"]].map(([l, k]) => row(l, base.filter(s => s.pl.status === k)))]];
  return `<details class="more"><summary>Estimated speed breakdowns <span class="s">median whole mph · usable estimates only</span></summary><div class="body tablewrap">
    <p class="note" style="margin-top:0">All ${esc(whoNoun())}'s shots${ST.stroke || ST.incoming ? " with the stroke / incoming filters" : ""}, any situation. The selected group's speed is in the panel above. Speeds from the older speed module are not mixed in.</p>
    <table class="t"><tr><th>Group</th><th class="r">Shots</th><th class="r">Usable speed</th><th class="r">Median</th><th class="r">Middle half</th><th></th></tr>
    ${rows.map(([h, r]) => `<tr><th colspan="6" style="background:var(--surface-2)">${h}</th></tr>${r.join("")}`).join("")}</table></div></details>`;
}
function pairsDetails(shots) {
  pairSets = { landing: {}, interception: {} };
  const tab = (key) => {
    const ok = shots.filter(s => s.pl.origin && s.pl.origin.zone !== "boundary" && s.pl.origin.half === "own" && PB.locOf(s, key) && PB.locOf(s, key).zone !== "boundary" && PB.locOf(s, key).half === "opponent");
    for (const s of ok) { const k = `${s.pl.origin.zone} → ${PB.locOf(s, key).zone}`; (pairSets[key][k] = pairSets[key][k] || []).push(s); }
    const top = Object.entries(pairSets[key]).sort((a, b) => b[1].length - a[1].length).slice(0, 6);
    return top.length ? `<table class="t" style="margin-bottom:10px"><tr><th>Hit from → ${key === "landing" ? "observed landing" : "interception"}</th><th class="r">Shots</th><th></th></tr>${top.map(([k, a]) => `<tr><td>${esc(k)}</td><td class="r">${a.length}</td><td><button type="button" class="btn sm" data-pair="${key}::${esc(k)}">Watch</button></td></tr>`).join("")}</table>` : "";
  };
  const a = tab("landing"), b = tab("interception");
  return `<details class="more"><summary>From where to where <span class="s">origin → outcome, both zones certain</span></summary><div class="body tablewrap">${a}${b}${!a && !b ? '<p class="note">Not enough shots with both zones certain.</p>' : '<p class="note" style="margin:0">Uses the matching shots (situation and filters above). Shots with a boundary or unknown location are left out of these pairs.</p>'}</div></details>`;
}

// ------------------------------------------------------------------ movement
function renderMovement() {
  const sec = $("#sec-movement");
  if (SEL.kind === "player") {
    const p = DB && DB.players[SEL.key];
    if (!p) {
      sec.innerHTML = `<div class="head"><h2 id="h-movement">Movement</h2><p>${esc(SEL.key)} · individual view</p></div>
        <div class="state-empty">No named positions for ${esc(SEL.key)} in this match, so movement is only available for the team. <button type="button" class="linkbtn" id="toTeam">Switch to ${esc(SEL.side)} (team)</button></div>`;
      $("#toTeam").onclick = () => go({ p: SEL.side });
      return;
    }
    const mv = p.movement, po = p.position, t2 = (l, v, n, x = "") => `<div class="mtile"><div class="l">${l}</div><div class="v">${v}</div><div class="n">${n}</div>${x}</div>`;
    sec.innerHTML = `<div class="head"><h2 id="h-movement">Movement</h2><p>${esc(SEL.key)} · from this player's named positions (about ±1 ft) in ${p.rallies} rallies.</p></div>
      <div class="panel" id="mvSlotWrap" hidden><div id="mvSlot"></div></div>
      <div class="panel"><h3>Running and position <span class="badge est">estimated</span></h3><p class="sub">Same method as singles; not checked against GPS. Compare with the partner in <button type="button" class="linkbtn" id="toPart">Partnership</button>.</p>
        <div class="mtiles">${t2("Distance per rally", `${mv.dist_per_rally_ft ?? "–"}<small> ft</small>`, `${mv.rallies} rallies`)}
          ${t2("Typical top speed", `${mv.typical_peak_mph ?? "–"}<small> mph</small>`, "per rally (median)")}
          ${t2("Fastest sprint", `${mv.top_sprint_mph ?? "–"}<small> mph</small>`, "held for 0.5 s", mv.top_sprints.length ? `<button type="button" class="btn sm" id="wSprint">Watch ${mv.top_sprints.length} sprints</button>` : "")}
          ${t2(`Up (within ${DB.up_ft} ft of the net)`, `${po.up_pct ?? "–"}%`, "of rally time")}
          ${t2("Average distance from the net", `${po.avg_depth_ft ?? "–"}<small> ft</small>`, `back (beyond ${DB.back_ft} ft) ${po.back_pct ?? "–"}% of rally time`)}
          ${t2("Played on the left half", `${po.left_pct ?? "–"}%`, "of rally time (their own left)")}</div></div>
      <div class="cols" style="margin-top:16px"><div class="panel"><h3>Court coverage</h3><p class="sub">Own half seen from behind ${esc(SEL.key)} (net at the top, left = their left). Darker = more time.</p>${heatSVG(Object.assign({}, p.heatmap, { fromLeft: true }))}</div>
        <div class="panel"><h3>By rally role</h3><p class="sub">Got within ${DB.up_ft} ft of the net / rallies in the role · median seconds after the serve · distance per rally.</p><table class="t">${ROLE_ROWS.map(([k, l]) => { const a = p.by_rally_role[k] || {}; return `<tr><td>${l}</td><td class="r">${a.n ? `${a.reached}/${a.n} · ${a.median_s ?? "–"} s · ${a.dist_per_rally_ft ?? "–"} ft` : "–"}</td></tr>`; }).join("")}</table></div></div>`;
    $("#toPart").onclick = () => go({ sec: "partnership" });
    const b = $("#wSprint", sec); if (b) b.onclick = () => { $("#mvSlotWrap").hidden = false; watchCustom($("#mvSlot"), momentItems(mv.top_sprints), `${SEL.key}: fastest sprints`, `mv|sprint|${SEL.key}`); };
    if (Player.key.startsWith("mv|")) { Player.mount($("#mvSlot")); $("#mvSlotWrap").hidden = false; }
    return;
  }
  const P = R.players[SEL.side], rn = P.running || {}, nt = P.net || {}, mv = P.movement || {}, pr = P.pressure || {}, fc = pr.faced || {}, ap = pr.applied || {};
  const dbl = R.type === "doubles";
  const tile = (l, v, n, extra = "") => `<div class="mtile"><div class="l">${l}</div><div class="v">${v}</div><div class="n">${n}</div>${extra}</div>`;
  const pc = v => v == null ? "–" : v + "%";
  const wr = winnersReliable();
  sec.innerHTML = `<div class="head"><h2 id="h-movement">Movement</h2><p>${esc(whoNoun())} · from player positions (about ±1 ft), 10 times a second${dbl ? " · team numbers: both partners" : ""}.</p></div>
    <div class="panel" id="mvSlotWrap" hidden><div id="mvSlot"></div></div>
    <div class="panel"><h3>Running <span class="badge est">estimated</span></h3><p class="sub">Not checked against GPS - compare players within the same match.${dbl ? " Doubles: distance is the team average of both partners; the sprint is the faster partner." : ""}</p>
      ${rn.rallies ? `<div class="mtiles">
        ${tile("Distance per rally", `${rn.dist_per_rally_ft}<small> ft</small>`, `${rn.total_dist_ft} ft in ${rn.rallies} rallies`)}
        ${tile("Typical top speed", `${rn.typical_peak_mph}<small> mph</small>`, "per rally (median)")}
        ${tile("Fastest sprint", `${rn.top_sprint_mph}<small> mph</small>`, "held for 0.5 s", (rn.top_sprints || []).length ? `<button type="button" class="btn sm" id="wSprint">Watch ${rn.top_sprints.length} sprint${rn.top_sprints.length > 1 ? "s" : ""}</button>` : "")}
        ${tile("Time above 6 mph", `${rn.sprint_time_pct}%`, "of rally time")}
        ${mv.avg_depth_ft != null ? tile("Average distance from the net", `${mv.avg_depth_ft}<small> ft</small>`, `at the kitchen line (within ${mv.kitchen_ft ?? 9} ft of the net) ${mv.kitchen_time_pct}% of rally time`) : ""}
        ${nt.rallies && !dbl ? tile("Reached the net", `${nt.reached_pct}%`, `of ${nt.rallies} rallies · median ${nt.median_arrival_s ?? "–"} s after the serve`) : ""}
        ${nt.n_contested && !dbl ? tile("First to the net", pc(nt.first_pct), `of ${nt.n_contested} rallies where someone got there${wr && nt.won_when_first_pct != null ? ` · won ${nt.won_when_first_pct}% when first, ${nt.won_when_opp_first_pct ?? "–"}% when not (rally winners from tracking)` : ""}`) : ""}
      </div>` : `<div class="state-empty">No movement data for this ${dbl ? "team" : "player"}.</div>`}</div>
    <div class="cols" style="margin-top:16px">
      <div class="panel"><h3>Court coverage</h3><p class="sub">Where ${dbl ? "the team's players" : "they"} stood during rallies, own half seen from behind ${dbl ? "them" : "the player"} (net at the top, left = their left). Darker = more time.</p>${heatSVG(mv.heatmap)}</div>
      <div class="panel"><h3>Time pressure <span class="badge est">estimated</span></h3><p class="sub">Time to spare on each shot: time since the opponent's hit minus 0.25 s to react and the time to cover the distance moved (at 12 ft/s). Under 0.3 s = rushed.</p>
        ${fc.n ? `<div class="mtiles">
          ${tile("Rushed on", pc(fc.rushed_pct), `of ${fc.n} shots · median ${fc.median_spare_s} s to spare`, (fc.rushed_rallies || []).length ? `<button type="button" class="btn sm" id="wRushed">Watch rushed rallies</button>` : "")}
          ${tile("Rushes the opponent", pc(ap.rushed_opponent_pct), `of ${ap.n} shots`, (ap.rushed_rallies || []).length ? `<button type="button" class="btn sm" id="wRushes">Watch</button>` : "")}
          ${wr ? tile("Errors when rushed", pc(fc.error_when_rushed_pct), `vs ${pc(fc.error_otherwise_pct)} otherwise · ${fc.n_rushed} rushed${fc.n_rushed < 10 ? " · limited sample" : ""} · error = last shot of a lost rally`) : ""}
        </div>` : `<div class="state-empty">No time-pressure data.</div>`}
        <p class="note" style="margin-bottom:0">Hit timing is exact to one video frame (0.03 s); positions about ±1 ft. Checked as a whole: rushed shots ended as the hitter's error 22% of the time vs 15% otherwise (2 matches).</p></div>
    </div>`;
  const slot = $("#mvSlot"), wrap = $("#mvSlotWrap");
  if (Player.key.startsWith("mv|")) { Player.mount(slot); wrap.hidden = false; }
  const w = (id, items, title, key) => { const b = $(id, sec); if (b) b.onclick = () => { wrap.hidden = false; watchCustom(slot, items, title, key); }; };
  w("#wSprint", momentItems(rn.top_sprints || []), "Fastest sprints (from 2 s before)", `mv|sprint|${SEL.key}`);
  w("#wRushed", rallyItems((fc.rushed_rallies || []).slice(0, 12)), "Rallies where they were rushed", `mv|rushed|${SEL.key}`);
  w("#wRushes", rallyItems((ap.rushed_rallies || []).slice(0, 12)), "Rallies where they rushed the opponent", `mv|rushes|${SEL.key}`);
}
function heatSVG(hm) {
  if (!hm || !hm.grid) return `<div class="state-empty">No position data.</div>`;
  const s = 11, c = hm.cell_ft, g = hm.grid, max = Math.max(1, ...g.flat()), mx = 3, top = 24, dMax = 29;
  const X = u => 66 + (mx + u) * s, Y = d => top + (1.5 + d) * s, W = 66 + (20 + 2 * mx) * s + 6, H = Y(dMax) + 26;
  let o = `<defs><clipPath id="heatClip"><rect x="${X(-mx)}" y="${Y(-1.5)}" width="${(20 + 2 * mx) * s}" height="${(dMax + 1.5) * s}" rx="6"/></clipPath></defs><rect x="${X(-mx)}" y="${Y(-1.5)}" width="${(20 + 2 * mx) * s}" height="${(dMax + 1.5) * s}" rx="6" fill="#e3eae4"/><rect x="${X(0)}" y="${Y(0)}" width="${20 * s}" height="${22 * s}" fill="var(--zs-0)"/>`;
  // grid columns run from the player's RIGHT (report._own_half); draw from behind them, so mirror: u = 20 - x
  o += `<g clip-path="url(#heatClip)">`;
  g.forEach((row, i) => row.forEach((v, j) => {
    if (!v) return;
    const d0 = i * c, x0 = -6 + j * c, u0 = hm.fromLeft ? x0 : 20 - (x0 + c);
    if (d0 > dMax || u0 + c < -mx || u0 > 20 + mx) return;
    const f = v / max, bin = f < .1 ? 1 : f < .3 ? 2 : f < .6 ? 3 : 4;
    o += `<rect x="${X(u0) + .5}" y="${Y(d0) + .5}" width="${c * s - 1}" height="${Math.min(c, dMax - d0) * s - 1}" fill="var(--zs-${bin})"/>`;
  }));
  o += `</g>`;
  const L = (u0, d0, u1, d1, w) => `<line x1="${X(u0)}" y1="${Y(d0)}" x2="${X(u1)}" y2="${Y(d1)}" stroke="#1f4d32" stroke-width="${w}"/>`;
  o += L(0, 0, 0, 22, 2) + L(20, 0, 20, 22, 2) + L(0, 22, 20, 22, 2) + L(0, 7, 20, 7, 2) + L(10, 7, 10, 22, 1);
  o += `<line x1="${X(-mx)}" y1="${Y(0)}" x2="${X(20 + mx)}" y2="${Y(0)}" stroke="var(--paddle)" stroke-width="4"/><text x="${X(10)}" y="${top - 8}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">net ↑</text>`;
  o += `<text x="${X(0)}" y="${H - 8}" font-size="12" font-weight="600" fill="var(--paddle)">◀ their left</text><text x="${X(20)}" y="${H - 8}" text-anchor="end" font-size="12" font-weight="600" fill="var(--paddle)">their right ▶</text>`;
  o += `<text x="${X(-mx) - 6}" y="${Y(3.5) + 4}" text-anchor="end" font-size="12" fill="var(--text-secondary)">kitchen</text><text x="${X(-mx) - 6}" y="${Y(22) + 4}" text-anchor="end" font-size="12" fill="var(--text-secondary)">baseline</text>`;
  return `<svg class="court" style="max-width:380px" viewBox="0 0 ${W} ${H}" role="img" aria-label="Court coverage heatmap">${o}</svg>
    <div class="legend"><span><span class="sw" style="background:var(--zs-1)"></span>little time</span><span><span class="sw" style="background:var(--zs-4)"></span>most time</span><span>2 ft cells, relative to this ${R.type === "doubles" ? "team" : "player"}'s busiest cell</span></div>`;
}

// ------------------------------------------------------------------ partnership (doubles; pickleball/doubles_players.py)
const FORM = [["both_up", "Both up", "var(--court-green)"], ["one_up", "One up, one back", "var(--zs-2)"], ["both_back", "Both back", "var(--surface-3)"]];
const ROLE_ROWS = [["server", "Serving"], ["server_partner", "Server's partner"], ["returner", "Returning"], ["returner_partner", "Returner's partner"]];
function formBar(f, label) {
  if (!f) return "";
  return `<div style="margin:6px 0 10px"><div class="note" style="margin-bottom:3px">${label}</div>
    <div style="display:flex;height:26px;border-radius:6px;overflow:hidden;border:1px solid var(--border)" role="img" aria-label="${esc(label)}: ${FORM.map(([k, l]) => `${l} ${f[k] ?? 0}%`).join(", ")}">
    ${FORM.filter(([k]) => f[k]).map(([k, l, c]) => `<div style="flex:${f[k]};background:${c};color:${k === "both_up" ? "#fff" : "var(--paddle)"};font-size:12px;font-weight:650;display:flex;align-items:center;justify-content:center;min-width:0;overflow:hidden;white-space:nowrap">${f[k] >= 9 ? f[k] + "%" : ""}</div>`).join("")}</div>
    <div class="legend" style="margin-top:4px">${FORM.map(([k, l, c]) => `<span><span class="sw" style="background:${c}"></span>${l} ${f[k] ?? 0}%</span>`).join("")}</div></div>`;
}
function renderPartnership() {
  const sec = $("#sec-partnership"), team = SEL.side, T0 = DB.teams[team] || {}, opp = PB.sidesOf(R).find(x => x !== team), T1 = DB.teams[opp] || {};
  const names = (R.teams[team] || []).filter(n => DB.players[n]);
  const pc = v => v == null ? "–" : v + "%";
  const card = n => {
    const p = DB.players[n], sh = p.shots, mv = p.movement, po = p.position, me = SEL.kind === "player" && SEL.key === n;
    return `<div class="panel" style="${me ? "border-color:var(--court-green);box-shadow:0 0 0 2px var(--lime-bg)" : ""}">
      <h3 style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap">${esc(n)} <button type="button" class="btn sm" data-ind="${esc(n)}">Individual placement →</button></h3>
      <p class="sub" style="margin:0 0 8px">${p.rallies} rallies with named positions</p>
      <table class="t"><tr><td>Shots credited</td><td class="r"><b>${sh.total}</b> · ${pc(sh.team_share_pct)} of the team's attributed shots</td></tr>
        <tr><td>Rally shots (3rd shot on)</td><td class="r"><b>${sh.rally_shots}</b> · ${pc(sh.team_rally_share_pct)}</td></tr>
        <tr><td>Middle balls taken</td><td class="r"><b>${p.middle.taken}</b> of the team's ${T0.middle ? T0.middle.n : "–"}</td></tr>
        <tr><td>Crossed to the partner's side to hit</td><td class="r"><b>${p.cross.n}</b> (${p.cross.volleys} volleys)</td></tr>
        <tr><td>Played on the left</td><td class="r"><b>${pc(po.left_pct)}</b> of rally time</td></tr>
        <tr><td>Up (within ${DB.up_ft} ft of the net)</td><td class="r"><b>${pc(po.up_pct)}</b> of rally time</td></tr>
        <tr><td>Average distance from the net</td><td class="r"><b>${po.avg_depth_ft ?? "–"}</b> ft</td></tr>
        <tr><td>Distance run per rally</td><td class="r"><b>${mv.dist_per_rally_ft ?? "–"}</b> ft · top sprint ${mv.top_sprint_mph ?? "–"} mph</td></tr></table>
      <div class="selactions" style="margin-top:8px">${p.middle.clips.length ? `<button type="button" class="btn sm" data-mom="mid:${esc(n)}">Watch middle balls (${p.middle.clips.length})</button>` : ""}${p.cross.clips.length ? `<button type="button" class="btn sm" data-mom="cross:${esc(n)}">Watch cross-overs (${p.cross.clips.length})</button>` : ""}</div></div>`;
  };
  const roleRows = ROLE_ROWS.map(([k, l]) => `<tr><td>${l}</td>${names.map(n => { const a = DB.players[n].by_rally_role[k] || {}; return `<td class="r">${a.n ? `${a.reached}/${a.n} up${a.median_s != null ? ` · ${a.median_s} s` : ""} · ${a.dist_per_rally_ft ?? "–"} ft` : "–"}</td>`; }).join("")}</tr>`).join("");
  const tm = (t, k) => { const v = (DB.teams[t] || {})[k]; return v; };
  const cmp = (label, f) => `<tr><td>${label}</td><td class="r">${f(T0) ?? "–"}</td><td class="r">${f(T1) ?? "–"}</td></tr>`;
  const ba = (T, k) => T.both_up_after_serve && T.both_up_after_serve[k] ? `${T.both_up_after_serve[k].reached}/${T.both_up_after_serve[k].n}${T.both_up_after_serve[k].median_s != null ? ` · ${T.both_up_after_serve[k].median_s} s` : ""}` : null;
  const left = Object.entries(T0.left_side_pct || {}).sort((a, b) => b[1] - a[1]);
  const meth = DB.hitter_methods || {}, nm = Object.values(meth).reduce((a, b) => a + b, 0);
  sec.innerHTML = `<div class="head"><h2 id="h-partnership">Partnership</h2><p>${esc(team)} · each player and the pair together, from named player positions (about ±1 ft) and the shots credited to each player.</p></div>
    <div class="panel" id="ptSlotWrap" hidden><div id="ptSlot"></div></div>
    <div class="cols">${names.map(card).join("")}</div>
    ${names.length > 2 ? `<p class="note">Team event: players rotate, so the pair numbers below combine every pair this team put on court.</p>` : ""}
    <div class="cols" style="margin-top:16px">
      <div class="panel"><h3>Together on court <span class="badge est">estimated</span></h3>
        <p class="sub">“Up” = within ${DB.up_ft} ft of the net. Measured on ${T0.pair_rallies} rallies (${T0.pair_seconds} s) where both partners were named.</p>
        ${formBar(T0.formation_time_pct, "Rally time")}${formBar(T0.formation_at_opp_contact, `When the opponents hit (${(T0.formation_at_opp_contact || {}).n || 0} contacts)`)}
        <table class="t"><tr><td>Distance between partners (median)</td><td class="r"><b>${T0.spacing.median_ft ?? "–"}</b> ft · ${T0.spacing.lateral_ft ?? "–"} ft sideways, ${T0.spacing.depth_gap_ft ?? "–"} ft in depth</td></tr>
          <tr><td>Who plays left</td><td class="r">${left.map(([n, v]) => `${esc(n)} ${v}%`).join(" · ") || "–"}</td></tr>
          <tr><td>Rallies with a side switch</td><td class="r"><b>${T0.switches.rallies}</b> of ${T0.switches.of}${T0.switches.clips.length ? ` <button type="button" class="btn sm" data-mom="switch">Watch</button>` : ""}</td></tr>
          <tr><td>Middle balls (contact in the centre third)</td><td class="r">${Object.entries(T0.middle.by_player).map(([n, v]) => `${esc(n)} ${v}`).join(" · ") || "–"} <span class="note">of ${T0.middle.n}</span></td></tr></table></div>
      <div class="panel"><h3>Getting to the net after the serve <span class="badge est">estimated</span></h3>
        <p class="sub">Per rally role: rallies where the player got within ${DB.up_ft} ft of the net / rallies in that role · median seconds after the serve · distance run per rally. Roles come from who is credited with the serve and the return (see How these numbers are made).</p>
        <div class="tablewrap"><table class="t"><tr><th>Role</th>${names.map(n => `<th class="r">${esc(n)}</th>`).join("")}</tr>${roleRows}</table></div>
        <p class="note">Both partners up after the serve: ${ba(T0, "receiving") ?? "–"} rallies when receiving, ${ba(T0, "serving") ?? "–"} when serving.</p></div>
    </div>
    <div class="panel" style="margin-top:16px"><h3>Pair against pair</h3><div class="tablewrap"><table class="t"><tr><th>Measure</th><th class="r">${esc(team)}</th><th class="r">${esc(opp || "Opponents")}</th></tr>
      ${cmp("Both up, share of rally time", T => T.formation_time_pct ? pc(T.formation_time_pct.both_up) : null)}
      ${cmp("Both back, share of rally time", T => T.formation_time_pct ? pc(T.formation_time_pct.both_back) : null)}
      ${cmp("Both up when the opponents hit", T => T.formation_at_opp_contact ? pc(T.formation_at_opp_contact.both_up) : null)}
      ${cmp("Partner distance (median)", T => T.spacing ? `${T.spacing.median_ft} ft` : null)}
      ${cmp("Both up after the serve, receiving", T => ba(T, "receiving"))}
      ${cmp("Both up after the serve, serving", T => ba(T, "serving"))}
      ${cmp("Rallies with a side switch", T => T.switches ? `${T.switches.rallies} of ${T.switches.of}` : null)}</table></div></div>
    <details class="more" style="margin-top:16px"><summary>How these numbers are made <span class="s">identity, hitter attribution, limits</span></summary><div class="body note">
      Players are named on ${DB.named_coverage_pct}% of tracked player frames (each player marked once, then followed by look; checked on the women's doubles: 98% of appearances right). Unnamed frames are left out.
      Who hit each shot: ${Object.entries(meth).map(([k, v]) => `${{ ball: "nearest to the ball in the video", depth: "deeper receiver (returns)", overlay: "server named by the score overlay", "only player seen": "only partner in view", unattributed: "not attributed (team-level only)" }[k] || k} ${v}`).join(" · ")} (of ${nm}).
      Checked against the 3D ball contact point where it is decisive: ball-based picks agree on 93% (152 of 163); overlay / return-rule picks on 67–70%; the previous method agreed on 57%. ${DB.hitters_changed} shots in this match changed hitter. No hand-tagged hitters exist yet for doubles.
      ${!DB.depth ? "No depth correction was applied (no camera calibration)." : `Positions carry the same camera depth correction as every other position number (the foot point is treated as ${DB.depth.h_ft} ft above the floor; the low camera otherwise reads the far end too deep). ` + (String(DB.depth.method).startsWith("rig") ? `One value for this court and camera (${esc(DB.depth.rig || "rig")}), pooled over all its matches: players waiting at the kitchen line at the serve, and servers behind the baseline, must measure the same at both ends.` : DB.depth.method === "kitchen line" ? "Calibrated on the kitchen-line rule: at the serve the returner's partner waits at the kitchen line at both ends." : DB.depth.method === "rules" ? "Calibrated on the server-stance rule: servers stand equally far behind the baseline at both ends." : "Default height: no rule could calibrate it for this match.")} “Up” = within ${DB.up_ft} ft of the net: measured positions pile up from the kitchen line (7 ft) to about 11 ft, and this line misclassifies the least rally time.
      Rally winners are not used here (unreliable in doubles).</div></details>`;
  $$("[data-ind]", sec).forEach(b => b.onclick = () => go({ p: b.dataset.ind, sec: "placement", zone: "" }));
  const slot = $("#ptSlot"), wrap = $("#ptSlotWrap");
  if (Player.key.startsWith("pt|")) { Player.mount(slot); wrap.hidden = false; }
  const mom = arr => arr.map(([rid, t]) => ({ rally: rid, t }));
  $$("[data-mom]", sec).forEach(b => b.onclick = () => {
    const [k, n] = b.dataset.mom.split(":");
    const list = k === "switch" ? T0.switches.clips : k === "mid" ? DB.players[n].middle.clips : DB.players[n].cross.clips;
    const title = k === "switch" ? `${team}: side switches` : k === "mid" ? `${n}: middle balls` : `${n}: crossed to the partner's side`;
    wrap.hidden = false; watchCustom(slot, momentItems(mom(list)), title, `pt|${b.dataset.mom}|${team}`);
  });
}

// ------------------------------------------------------------------ clips
function renderClips() {
  const sec = $("#sec-clips"), { list, agg } = selectionModel(), rallies = R.rallies;
  const mode = ST.list === "rallies" ? "rallies" : "shots";
  sec.innerHTML = `<div class="head"><h2 id="h-clips">Clips</h2><p>Watch the shots behind the current selection, or whole rallies. Nothing plays until you press a clip.</p></div>
    <div class="filterbar"><div class="seg" role="group" aria-label="Clip list">
      <button type="button" data-list="" aria-pressed="${mode === "shots"}">Shots in current selection <span class="n">${list.length}</span></button>
      <button type="button" data-list="rallies" aria-pressed="${mode === "rallies"}">Whole rallies <span class="n">${rallies.length}</span></button></div>
      <span class="note">${mode === "shots" ? `${esc(selectionLabel())} · ${esc(whoNoun())} · <button type="button" class="linkbtn" id="toPl">change in Placement &amp; speed</button>` : `All ${rallies.length} rallies found in the video, in order.${R.type === "doubles" ? " Winners are from tracking and unreliable in doubles." : ""}`}</span></div>
    <div class="panel" id="clipSlot"></div>`;
  $$("[data-list]", sec).forEach(b => b.onclick = () => go({ list: b.dataset.list }));
  $("#toPl", sec) && ($("#toPl", sec).onclick = () => go({ sec: "placement" }));
  Player.mount($("#clipSlot"));
  if (mode === "shots") { const k = selKey(); if (Player.key !== k) Player.load(shotItems(list, ST.view), { key: k, title: selectionLabel(), idx: ST.clip ? +ST.clip - 1 : -1 }); else Player.refresh(); }
  else if (Player.key !== "rallies") Player.load(rallies.map(r => ({ kind: "rally", r })), { key: "rallies", title: "Whole rallies" });
  else Player.refresh();
}

// ------------------------------------------------------------------ advanced analysis
function renderAdvanced() {
  const sec = $("#sec-advanced"), P = R.players[SEL.side] || {}, ind = SEL.kind === "player" ? (R.individuals || {})[SEL.key] : null;
  const T2 = P.tactics || {}, dbl = R.type === "doubles";
  const bars = (entries, unit = "") => { const max = Math.max(1, ...entries.map(e => e[1])); return entries.map(([k, v, x]) => `<div style="display:grid;grid-template-columns:minmax(90px,150px) 1fr 70px;gap:10px;align-items:center;margin:5px 0;font-size:13.5px"><span>${esc(cap(k))}</span><span style="background:var(--surface-3);height:10px;border-radius:5px;overflow:hidden"><span style="display:block;height:100%;width:${100 * v / max}%;background:var(--series-1)"></span></span><span class="tnum" style="text-align:right">${v}${unit}${x || ""}</span></div>`).join(""); };
  const mixInd = () => { const c = {}; PB.shotsFor(R, SEL).forEach(x => { if (x.shot) c[x.shot] = (c[x.shot] || 0) + 1; }); return c; };
  const mix = Object.entries((ind ? mixInd() : P.shot_mix) || {}).sort((a, b) => b[1] - a[1]);
  const third = Object.entries(P.third_shot || {}).sort((a, b) => b[1].count - a[1].count);
  const st = T2.strokes || {}, segN = { pulled_wide: "Pulled wide", near_body: "Near the body", ran_around: "Ran around" };
  const resp = Object.entries(T2.response || {}).filter(([, v]) => v.n >= 4);
  const sp = P.spin || {};
  const wr = winnersReliable();
  const projected = PB.shotsFor(R, SEL).filter(s => s.pl.status === "intercepted").length;
  sec.innerHTML = `<div class="head"><h2 id="h-advanced">Advanced analysis</h2><p>Secondary and experimental measures. Read each one with its evidence label; none of these feed the Overview observations.</p></div>
    <div class="panel" id="advSlotWrap" hidden><div id="advSlot"></div></div>
    <details class="more" open><summary>Shot-type analysis <span class="s"><span class="badge auto">automatic labels</span> drives / dinks mostly right; drops 19%, resets 4% found</span></summary><div class="body">
      ${ind ? `<p class="note" style="margin-top:0">Individual view: only the shot mix is available per player; the rest of this block is team-level - switch to ${esc(SEL.side)} (team) to see it.</p>` : ""}
      <div class="cols">
        <div><h3 style="margin:0 0 6px;font-size:14px">Shot mix</h3>${mix.length ? bars(mix) : '<p class="note">No typed shots.</p>'}</div>
        ${!ind && third.length ? `<div><h3 style="margin:0 0 6px;font-size:14px">Third shot by type</h3><table class="t"><tr><th>Type</th><th class="r">Count</th>${wr ? '<th class="r">Rallies won</th>' : ""}<th></th></tr>${third.map(([k, v]) => `<tr><td>${esc(cap(k))}</td><td class="r">${v.count}</td>${wr ? `<td class="r">${v.won_pct ?? "–"}%</td>` : ""}<td>${(v.rallies || []).length ? `<button type="button" class="btn sm" data-rl="${(v.rallies || []).join(",")}" data-title="Third shot: ${esc(k)}">Watch</button>` : ""}</td></tr>`).join("")}</table>${wr ? '<p class="note">Rallies won: from tracked winners.</p>' : ""}</div>` : ""}
        ${!ind && (st.forehand || st.backhand) ? `<div><h3 style="margin:0 0 6px;font-size:14px">Forehand &amp; backhand: forced or chosen?</h3><p class="note" style="margin-top:0">Move before the stroke: pulled wide = 3+ ft toward that side; near the body = little movement; ran around = moved the other way first. <span class="badge est">estimated</span></p>
          <table class="t"><tr><th>Stroke</th>${Object.values(segN).map(v => `<th class="r">${v}</th>`).join("")}<th></th></tr>${["forehand", "backhand"].filter(k => st[k]).map(k => { const c = st[k], n = Object.values(c).reduce((a, b) => a + b, 0); return `<tr><td>${cap(k)} <span class="note">(${n})</span></td>${Object.keys(segN).map(s => `<td class="r">${c[s] ? `${pct(c[s], n)}%` : "–"}</td>`).join("")}<td>${((T2.stroke_rallies || {})[`${k}:pulled_wide`] || []).length ? `<button type="button" class="btn sm" data-rl="${T2.stroke_rallies[`${k}:pulled_wide`].join(",")}" data-title="${cap(k)}s after being pulled wide">Pulled wide</button>` : ""}</td></tr>`; }).join("")}</table></div>` : ""}
        ${!ind && resp.length ? `<div><h3 style="margin:0 0 6px;font-size:14px">Response to the opponent's shot type</h3><table class="t"><tr><th>Opponent's shot</th><th class="r">Shots</th><th class="r">Moved</th><th class="r">Rushed</th><th></th></tr>${resp.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="r">${v.n}</td><td class="r">${v.moved_ft} ft</td><td class="r">${v.rushed_pct ?? "–"}%</td><td><button type="button" class="btn sm" data-rl="${v.rallies.slice(0, 10).join(",")}" data-title="After the opponent's ${esc(k.toLowerCase())}">Watch</button></td></tr>`).join("")}</table></div>` : ""}
      </div></div></details>
    <details class="more"><summary>Win probability &amp; big points <span class="s"><span class="badge est">model output</span></span></summary><div class="body">${wpBlock()}</div></details>
    <details class="more"><summary>Spin &amp; net clearance <span class="s"><span class="badge exp">experimental</span> not checked against hand labels</span></summary><div class="body tablewrap">
      ${sp.overall ? `<table class="t"><tr><th>Shot</th><th class="r">Topspin</th><th class="r">Flat</th><th class="r">Slice</th><th class="r">Net clearance (typical)</th><th class="r">Shots</th></tr>${[...Object.entries(sp.groups || {}), ["All shots", sp.overall]].map(([k, v]) => `<tr><td>${esc(k)}</td><td class="r">${v.topspin}%</td><td class="r">${v.flat}%</td><td class="r">${v.slice}%</td><td class="r">${v.clear_ft != null ? v.clear_ft + " ft" : "–"}</td><td class="r">${v.n}</td></tr>`).join("")}</table>
        <p class="note">From the 3D flight model with spin (fits 94% of ball positions vs 72% without; serves come out as topspin as expected). The topspin / flat / slice cut-offs are first guesses; about 1 in 5 shots has no believable 3D flight and is left out.${ind ? " Team-level." : ""}</p>` : `<p class="note">No 3D flights for this match yet.</p>`}</div></details>
    <details class="more"><summary>Projected landings <span class="s"><span class="badge exp">not available</span></span></summary><div class="body note">Where an intercepted ball would have landed is <b>not shown</b>: no projection model has been validated yet (a holdout test on flights with known bounces is planned). ${projected} of ${esc(whoNoun())}'s shots were intercepted; their interception points are in Placement &amp; speed → Interceptions. Nothing is guessed from speed alone.</div></details>
    ${!dbl ? `<details class="more"><summary>Side-by-side match numbers <span class="s">both players, with how far each number can be trusted</span></summary><div class="body tablewrap">${h2hTable()}</div></details>` : ""}
    <div class="panel" style="margin-top:16px"><h3>More</h3><p class="sub" style="margin:0">${M.replay ? `<a class="btn sm" href="replay3d.html?match=${encodeURIComponent(M.replay)}">Open the 3D replay →</a> ` : ""}<a class="btn sm" href="hub.html?tab=accuracy">Accuracy of every metric →</a> <a class="btn sm ghost" href="../docs/PLACEMENT.md">Placement method (docs/PLACEMENT.md)</a></p></div>`;
  const slot = $("#advSlot"), wrap = $("#advSlotWrap");
  if (Player.key.startsWith("adv|")) { Player.mount(slot); wrap.hidden = false; }
  $$("[data-rl]", sec).forEach(b => b.onclick = () => { wrap.hidden = false; watchCustom(slot, rallyItems(b.dataset.rl.split(",").map(Number)), b.dataset.title, `adv|${b.dataset.title}|${SEL.key}`); });
  $$("#wpsvg [data-r]", sec).forEach(c => c.addEventListener("click", () => { wrap.hidden = false; watchCustom(slot, rallyItems([+c.dataset.r]), `Key moment · rally ${rno(+c.dataset.r)}`, `adv|wp${c.dataset.r}`); }));
}
function wpBlock() {
  const rs = R.rallies.filter(r => r.wp_before != null), info = R.win_probability || {};
  if (!rs.length) return `<p class="note">Not available for this match${R.type === "doubles" ? " (doubles scoring is reconstructed with too little certainty)" : ""}.</p>`;
  const me = info.player === SEL.side, val = r => 100 * (me ? r.wp_before : 1 - r.wp_before);
  const W = 720, H = 190, L = 36, Rm = 10, T0 = 12, B = 24, X = i => L + (W - L - Rm) * i / Math.max(1, rs.length - 1), Y = v => T0 + (H - T0 - B) * (1 - v / 100);
  const top = [...rs].sort((a, b) => b.importance - a.importance).slice(0, 6);
  let g = `<line x1="${L}" y1="${Y(50)}" x2="${W - Rm}" y2="${Y(50)}" stroke="var(--border-strong)" stroke-dasharray="3 3"/>` + [0, 50, 100].map(v => `<text x="${L - 6}" y="${Y(v) + 4}" font-size="11.5" text-anchor="end" fill="var(--text-muted)">${v}%</text>`).join("");
  rs.forEach((r, i) => { if (i && r.game !== rs[i - 1].game) g += `<line x1="${X(i - .5)}" y1="${T0}" x2="${X(i - .5)}" y2="${H - B}" stroke="var(--border)"/><text x="${X(i - .5) + 4}" y="${T0 + 10}" font-size="11.5" fill="var(--text-muted)">Game ${r.game}</text>`; });
  g += `<path d="${rs.map((r, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(val(r)).toFixed(1)}`).join("")}" fill="none" stroke="var(--series-${PB.sidesOf(R).indexOf(SEL.side) + 1})" stroke-width="2.2"/>`;
  rs.forEach((r, i) => { if (top.includes(r)) g += `<circle cx="${X(i)}" cy="${Y(val(r))}" r="6" fill="var(--ball)" stroke="var(--paddle)" stroke-width="1.4" data-r="${r.id}" style="cursor:pointer"><title>G${r.game} ${r.score_before} · ${r.server} serving · won by ${r.winner}</title></circle>`; });
  const bp = (R.players[SEL.side] || {}).big_points || {};
  return `<p class="note" style="margin-top:0">${esc(SEL.side)}'s chance of winning the match before each rally (side-out scoring model using the reconstructed score; inherits server 98% / winner 96% accuracy). Dots = the 6 rallies that moved the odds most - click to watch.</p>
    <svg id="wpsvg" viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px" role="img" aria-label="Win probability by rally">${g}<text x="${L}" y="${H - 6}" font-size="11.5" fill="var(--text-muted)">rallies in order →</text></svg>
    ${bp.n_big ? `<p class="note">Big points (top quarter by importance): won ${bp.big_won_pct}% of ${bp.n_big} vs ${bp.rest_won_pct}% of the other ${bp.n_rest}${bp.n_big < 10 ? " · limited sample" : ""}. One match is too little to call this a trait.</p>` : ""}`;
}
function h2hTable() {
  const ACC = Object.fromEntries((window.ACCURACY || []).map(a => [a.metric, a])), names = PB.sidesOf(R), P = R.players;
  const tag = m => { const a = ACC[m]; return a ? `<span class="lvl ${a.level}">${LEVELS[a.level][0]}</span>` : ""; };
  const g = (o, ...k) => k.reduce((x, y) => x == null ? null : x[y], o);
  const spd = (n, role) => { const sp = PB.speedStats(PB.applyFilters(PB.shotsFor(R, { kind: "side", side: n, key: n }), { role }, "origin")); return sp.median == null ? null : `${sp.median} mph (${sp.usable})`; };
  const rows = [["Estimated speed, serves (median, usable n)", n => spd(n, "serve"), "Estimated speed"], ["Estimated speed, all shots", n => spd(n, "all"), "Estimated speed"],
    ...(winnersReliable() ? [["Rallies won (tracking)", n => g(P[n], "summary", "rallies_won_pct", "value") + "%", "Rally winner"], ["Won when serving", n => g(P[n], "summary", "won_on_serve_pct", "value") + "%", "Rally winner"], ["Won when returning", n => g(P[n], "summary", "won_on_return_pct", "value") + "%", "Rally winner"]] : []),
    ["Rushed on (under 0.3 s to spare)", n => (g(P[n], "pressure", "faced", "rushed_pct") ?? "–") + "%", "Time pressure (time to spare)"],
    ["Reached the net", n => (g(P[n], "net", "reached_pct") ?? "–") + "% of rallies", "Player court position"],
    ["Distance run per rally", n => (g(P[n], "running", "dist_per_rally_ft") ?? "–") + " ft", "Distance run / running speed"]];
  return `<table class="t"><tr><th>Measure</th>${names.map((n, i) => `<th class="r"><span class="dot" style="background:var(--series-${i + 1})"></span> ${esc(n)}</th>`).join("")}<th>Evidence</th></tr>
    ${rows.map(([l, f, m]) => `<tr><td>${l}</td>${names.map(n => `<td class="r">${f(n) ?? "–"}</td>`).join("")}<td>${tag(m)}</td></tr>`).join("")}</table>
    <p class="note">Speeds here use the placement speed definition only (launch speed off the paddle); the older speed module is not mixed in.</p>`;
}

// ------------------------------------------------------------------ data quality dialog
function openDQ() {
  const dq = $("#dq");
  if (!R) { dq.innerHTML = dqShell(`<p>No match report: only the processing status is shown for this match.</p>`); dq.showModal(); bindDQ(); return; }
  const all = PB.shotsFor(R, SEL), f = curFilters(), inferred = R.rallies.flatMap(r => r.shots).filter(s => s.inferred && (SEL.kind === "player" ? s.hitter === SEL.key : s.player === SEL.side)).length;
  const roleShots = PB.applyFilters(all, { role: ST.role }, "origin");
  const L = PB.aggregate(roleShots, "landing"), I = PB.aggregate(roleShots, "interception"), O = PB.aggregate(roleShots, "origin"), sp = PB.speedStats(roleShots);
  const q = { human: 0, auto2: 0, auto1: 0 }; L.recorded.forEach(s => q[PB.evidenceOf(s, "landing")]++);
  const prox = O.recorded.filter(s => PB.evidenceOf(s, "origin") === "proxy").length;
  const val = PB.venueValidation(M.tournament), st = L.status;
  dq.innerHTML = dqShell(`
    <p style="margin:0">${esc(whoNoun())} · ${ST.role === "all" ? "all shots" : esc(ROLE_NOUN[ST.role])} · this match. Every number below is for this scope; the filters under “More filters” are not applied here.</p>
    <div><h3>Coverage</h3><ul>
      <li><b>${roleShots.length}</b> analysed ${esc(ROLE_NOUN[ST.role])}${inferred ? ` (${inferred} more were deduced from the rules but not seen - they carry no measurement and are left out)` : ""}; ${R.rallies.length} rallies found in the video (${esc(M.check || "no external cross-check")}).</li>
      <li>Outgoing flight: <b>${st.bounced || 0}</b> bounced first · <b>${st.intercepted || 0}</b> intercepted · <b>${st.net || 0}</b> ended at the net · <b>${st.unresolved || 0}</b> unresolved (never forced into in / out / net).</li>
      <li>Landings: <b>${L.denominator}</b> recorded of ${L.population} that bounced first${L.missing.length ? ` (${L.missing.length} without a position)` : ""}. Interceptions: <b>${I.denominator}</b> recorded of ${I.population}. Contact points: <b>${O.denominator}</b>.</li></ul></div>
    <div><h3>Human-reviewed vs automatic</h3><ul>
      <li>Landings: <b>${q.human}</b> human-reviewed (hand-tagged bounces), <b>${q.auto2}</b> where two automatic sources agree, <b>${q.auto1}</b> from one automatic source.</li>
      <li>Contact points: <b>${O.denominator - prox}</b> ball position at contact (3D fit), <b>${prox}</b> the hitter's feet (position proxy - where the player stood, not the ball).</li></ul></div>
    <div><h3>Validation at this venue <span class="lvl ${val.level}">${LEVELS[val.level][0]}</span></h3><p style="margin:0;color:var(--text-secondary)">${esc(val.detail)}</p></div>
    <div><h3>Not in a zone, unknown, excluded</h3><ul>
      <li>Landings: ${L.boundary.length} on a zone boundary, ${L.outside.length} estimated outside the court${L.otherHalf.length ? `, ${L.otherHalf.length} on the wrong half` : ""} - all counted in the ${L.denominator}, none placed in a zone.</li>
      <li>Interceptions: ${I.boundary.length} on a zone boundary (interception positions are often uncertain), ${I.outside.length} outside, ${I.otherHalf.length} on the hitter's own half.</li>
      <li>Contact points: ${O.boundary.length} on a boundary, ${O.outside.length} wide of the sidelines, ${O.otherHalf.length} on the opponent's half.</li>
      <li>Unresolved shots (${st.unresolved || 0}) have no landing and no interception; they still count as matching shots.</li></ul></div>
    <div><h3>Speed</h3><ul><li>Estimated launch speed off the paddle (physics fit of the outgoing flight, before any opponent contact), whole mph. Usable for <b>${sp.usable}</b> of ${sp.n}: shown only when the flight is physically plausible and at least 6 ball positions were seen.</li>
      <li>Not checked against radar. The older speed module's values are not used on this page.</li></ul></div>
    <div><h3>Outcomes &amp; scoring</h3><ul><li>${R.type === "doubles" ? "Doubles rally winners from tracking are weak (49–78% right), so rally-level outcomes are not summarised; the official result comes from the match card." : `Rally winners from tracking: ${esc(M.check || "")}.${winnersReliable() ? "" : " Not summarised here because the cross-check is weak."}`}</li></ul></div>
    <div><h3>Definitions</h3><ul><li>Every location is seen from behind the hitter, looking over the net: left = the hitter's left.</li><li>Zones from the net on each half: kitchen 0–7 ft, midcourt 7–14.5 ft, deep 14.5–22 ft; left / centre / right thirds.</li><li>Landing = observed first bounce; interception = the opponent's contact before a bounce (never a landing); hit from = contact point.</li><li>Projected landings of intercepted balls are not available (no validated model).</li></ul></div>
    <p style="margin:0"><a href="hub.html?tab=accuracy">Accuracy of every metric →</a> · <a href="../docs/PLACEMENT.md">Placement method (docs/PLACEMENT.md)</a></p>`);
  dq.showModal(); bindDQ();
}
const dqShell = body => `<div class="dqh"><h2 id="dqTitle">Data quality</h2><button type="button" class="btn sm" id="dqClose" autofocus>Close</button></div><div class="dqb">${body}</div>`;
function bindDQ() { $("#dqClose").onclick = () => $("#dq").close(); $("#dq").onclick = e => { if (e.target.id === "dq") $("#dq").close(); }; }

// ------------------------------------------------------------------ site-wide accuracy page (hub.html without ?m)
function siteAccuracy() {
  $("#loading")?.remove();
  $("#navacc").classList.add("on");
  document.title = "Accuracy · Picklytics";
  $("#crumb").innerHTML = `<a href="matches.html">Tournaments</a><span aria-hidden="true">›</span><span class="here">Accuracy</span>`;
  $("#mh").innerHTML = `<div><h1>How accurate is each number?</h1><div class="who-line">What every metric on this site is worth, and how we know. Levels: ${Object.entries(LEVELS).map(([k, [l, d]]) => `<span><span class="lvl ${k}">${l}</span> ${d}</span>`).join("")}</div></div>`;
  const groups = [...new Set(ACCURACY.map(a => a.group))], sec = $("#sec-accuracy");
  sec.hidden = false;
  sec.innerHTML = `<div class="panel tablewrap"><table class="t"><tr><th>Metric</th><th>Accuracy</th><th>Level</th><th>How we know</th></tr>
    ${groups.map(gr => `<tr><th colspan="4" style="background:var(--surface-2);color:var(--paddle)">${esc(gr)}</th></tr>` + ACCURACY.filter(a => a.group === gr).map(a => `<tr><td><b>${esc(a.metric)}</b></td><td style="min-width:180px">${esc(a.range)}</td><td><span class="lvl ${a.level}">${LEVELS[a.level][0]}</span></td><td style="color:var(--text-secondary);font-size:13px">${esc(a.basis)}</td></tr>`).join("")).join("")}</table></div>
    <div class="panel"><h3>Placement validation by venue</h3><table class="t">${Object.entries(window.PLACEMENT_VALIDATION || {}).map(([k, v]) => `<tr><td><b>${esc(k === "_default" ? "Other venues" : ((J.tournaments || []).find(t => t.id === k) || { name: k }).name)}</b></td><td><span class="lvl ${v.level}">${LEVELS[v.level][0]}</span></td><td style="color:var(--text-secondary);font-size:13px">${esc(v.detail)}</td></tr>`).join("")}</table></div>`;
}

init();
})();
