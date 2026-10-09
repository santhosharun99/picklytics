// Tests for dashboard/js/coach-core.js (filters, aggregation, playlists, speed, clip timing, URL state,
// observations). Run: node --test tests/js/*.test.mjs   (also run from pytest via tests/test_dashboard_js.py)
import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
const require = createRequire(import.meta.url);
const PB = require("../../dashboard/js/coach-core.js");

// ---- synthetic report: one player, hand-built placement records ----
let fr = 0;
function shot(o) {
  const f = (fr += 60), t = f / 30;
  const loc = (zone, half, extra = {}) => ({ zone, half, inside: !/behind|wide/.test(zone), candidates: zone === "boundary" ? ["deep left", "deep centre"] : [zone], u: 3, d: 18, ...extra });
  const pl = { role: o.role || "later", incoming: o.incoming || "after bounce", status: o.status || "bounced", end_frame: f + 30,
    origin: loc(o.origin || "deep centre", o.originHalf || "own", { evidence: o.proxy ? "hitter's feet (player-position proxy)" : "ball position at contact (3D fit)" }),
    landing: o.status === "bounced" || !o.status ? (o.land === null ? { state: "unknown" } : { state: "observed", ...loc(o.land || "deep left", "opponent"), quality: o.q || "one automatic source", frame: f + 24 }) : { state: "not-applicable" },
    interception: o.status === "intercepted" ? { state: "observed", ...loc(o.icpt || "kitchen left", "opponent", { evidence: "ball position at the opponent's contact (3D fit)" }), frame: f + 20 } : { state: "not-applicable" },
    projected: { state: o.status === "intercepted" ? "unknown" : "not-applicable" },
    speed: o.mph === null ? { mph: null, why: "few positions" } : { mph: o.mph ?? 40 } };
  return { t, f, player: o.player || "A", stroke: o.stroke || "forehand", shot: o.shotType || "drive", pl, inferred: !!o.inferred, hitter: o.hitter };
}
function report(shots, extra = {}) { return { type: extra.type, players: { A: {}, B: {} }, rallies: [{ id: 7, start_t: 0, end_t: 99, shots }], ...extra }; }
const sideA = { kind: "side", key: "A", side: "A" };

test("filters: role, stroke, incoming; status only in the Hit from view; evidence per view", () => {
  const R = report([shot({ role: "third" }), shot({ role: "third", stroke: "backhand" }), shot({ role: "serve", incoming: "serve" }),
    shot({ role: "third", status: "intercepted" }), shot({ role: "third", q: "human-reviewed" })]);
  const S = PB.shotsFor(R, sideA);
  assert.equal(PB.applyFilters(S, { role: "third" }, "landing").length, 4);
  assert.equal(PB.applyFilters(S, { role: "third", stroke: "backhand" }, "landing").length, 1);
  assert.equal(PB.applyFilters(S, { incoming: "serve" }, "landing").length, 1);
  // status is ignored outside Hit from (Landings / Interceptions ARE a status)
  assert.equal(PB.applyFilters(S, { status: "intercepted" }, "landing").length, 5);
  assert.equal(PB.applyFilters(S, { status: "intercepted" }, "origin").length, 1);
  assert.equal(PB.applyFilters(S, { ev: "human" }, "landing").length, 1);
  assert.equal(PB.applyFilters(S, { ev: "ball" }, "origin").length, 5);
});

test("inferred shots and other players are left out; doubles individual = named hitter", () => {
  const R = report([shot({}), shot({ inferred: true }), shot({ player: "B" }), shot({ hitter: "X" }), shot({ hitter: "Y" })], { type: "doubles" });
  assert.equal(PB.shotsFor(R, sideA).length, 3);
  assert.equal(PB.shotsFor(R, { kind: "player", key: "X", side: "A" }).length, 1);
});

test("aggregation: every recorded location is in exactly one group; landings and interceptions never mix", () => {
  const R = report([shot({ land: "deep left" }), shot({ land: "deep left" }), shot({ land: "kitchen right" }), shot({ land: "boundary" }),
    shot({ land: "behind left" }), shot({ land: "deep wide-right" }), shot({ land: null }), shot({ status: "intercepted" }), shot({ status: "net" }), shot({ status: "unresolved" })]);
  const S = PB.shotsFor(R, sideA), L = PB.aggregate(S, "landing"), I = PB.aggregate(S, "interception");
  const cells = Object.values(L.cells).reduce((a, l) => a + l.length, 0);
  assert.equal(L.denominator, 6);                      // observed landings only
  assert.equal(cells + L.boundary.length + L.outside.length + L.otherHalf.length, L.denominator);
  assert.equal(L.boundary.length, 1); assert.equal(L.outside.length, 2);   // behind / wide are never forced into a zone
  assert.equal(L.missing.length, 1);                   // bounced, position unknown: not in the denominator
  assert.equal(L.population, 7);
  assert.deepEqual(L.status, { bounced: 7, intercepted: 1, net: 1, unresolved: 1 });
  assert.equal(I.denominator, 1);
  const ids = new Set(L.recorded.map(s => s.t));
  assert.ok(I.recorded.every(s => !ids.has(s.t)), "an interception is never a landing");
});

test("selected zone = playlist: counts, row selection, not-in-a-zone groups", () => {
  const R = report([shot({ land: "deep left" }), shot({ land: "deep left" }), shot({ land: "deep right" }), shot({ land: "kitchen left" }), shot({ land: "boundary" })]);
  const a = PB.aggregate(PB.shotsFor(R, sideA), "landing");
  assert.equal(PB.selectShots(a, "deep left").length, 2);
  assert.equal(PB.selectShots(a, "band:deep").length, 3);
  assert.equal(PB.selectShots(a, "boundary").length, 1);
  assert.equal(PB.selectShots(a, "").length, a.denominator);
  assert.equal(PB.selectShots(a, "midcourt centre").length, 0);
  assert.ok(PB.validZone("behind left", "origin") && !PB.validZone("behind left", "landing"));
});

test("estimated speed: usable estimates only, whole mph, median", () => {
  const S = PB.shotsFor(report([shot({ mph: 30 }), shot({ mph: 41 }), shot({ mph: null }), shot({ mph: 50.4 })]), sideA);
  const sp = PB.speedStats(S);
  assert.equal(sp.n, 4); assert.equal(sp.usable, 3); assert.equal(sp.median, 41); assert.equal(sp.q1, null);   // quartiles only from 6 usable
  assert.deepEqual(PB.speedStats([]), { n: 0, usable: 0, median: null, q1: null, q3: null, max: null });
});

test("clip window: lead-in, follow-through past the event, video offset", () => {
  const s = shot({});                                   // landing 24 frames (0.8 s) after the hit
  const R = report([s], { video_offset: 10 });
  const w = PB.clipWindow(R, PB.shotsFor(R, sideA)[0], "landing");
  assert.ok(Math.abs(w.hit - (s.t - 10)) < 1e-9);
  assert.ok(Math.abs(w.start - (s.t - PB.LEAD_S - 10)) < 1e-9);
  assert.ok(Math.abs(w.event - (s.t + 0.8 - 10)) < 1e-6);
  assert.ok(Math.abs(w.end - (w.event + PB.FOLLOW_S)) < 1e-6);
  const r = PB.rallyWindow(R, R.rallies[0]);
  assert.equal(r.start, 0);                            // never before the start of the cut video
});

test("URL state: round trip, defaults dropped, legacy hashes, invalid values cleaned", () => {
  const st = PB.parseState("?m=x&sec=placement&p=Kate+Fahey&role=third&zone=deep+left");
  assert.equal(st.p, "Kate Fahey"); assert.equal(st.zone, "deep left"); assert.equal(st.view, "landing");
  assert.equal(PB.stateToSearch(st), "?m=x&sec=placement&p=Kate+Fahey&role=third&zone=deep+left");
  assert.equal(PB.parseState("?m=x", "#players").sec, "placement");
  assert.equal(PB.parseState("?m=x", "#rallies").list, "rallies");
  const bad = PB.parseState("?m=x&role=lob&view=nope&zone=behind+left&status=net&ev=human");
  assert.equal(bad.role, "all"); assert.equal(bad.view, "landing"); assert.equal(bad.zone, ""); assert.equal(bad.status, ""); assert.equal(bad.ev, "human");
  assert.equal(PB.parseState("?m=x&view=origin&ev=human").ev, "");     // evidence options are per view
});

test("switching view keeps situation filters, clears zone / evidence / status with a note", () => {
  const st = PB.parseState("?m=x&role=third&view=origin&status=intercepted&ev=proxy&zone=deep+left&stroke=backhand");
  const { state, notes } = PB.switchView(st, "interception");
  assert.equal(state.role, "third"); assert.equal(state.stroke, "backhand");
  assert.equal(state.zone, ""); assert.equal(state.status, ""); assert.equal(state.ev, "");
  assert.equal(notes.length, 2);
});

test("observations: none below the minimum sample, fewer rather than invented", () => {
  const few = report(Array.from({ length: 8 }, () => shot({ land: "deep left" })));
  assert.equal(PB.observations(few, sideA).length, 0);
  const many = report([...Array.from({ length: 12 }, () => shot({ land: "deep left" })), ...Array.from({ length: 4 }, () => shot({ land: "kitchen right" }))]);
  const obs = PB.observations(many, sideA);
  assert.ok(obs.length >= 1 && obs.length <= 3);
  const z = obs.find(o => o.zone === "deep left");
  assert.ok(z, "clear leading zone is reported");
  assert.match(z.measure, /^12 of 16 recorded landings \(75%\)$/);
  // a close runner-up is not called "the most common zone"
  const close = report([...Array.from({ length: 9 }, () => shot({ land: "deep left" })), ...Array.from({ length: 8 }, () => shot({ land: "deep right" }))]);
  assert.ok(!PB.observations(close, sideA).some(o => o.id.startsWith("landing:all:")));
  // and no observation rests on the shot-type label
  const typed = report(Array.from({ length: 30 }, (_, i) => shot({ land: i % 3 ? "deep left" : "deep right", shotType: "drop" })));
  assert.ok(PB.observations(typed, sideA).every(o => !/drop/i.test(o.headline)));
});

test("real report: observation counts equal the playlist behind them", { skip: !fs.existsSync("outputs/reports/boyer_devilliers_full.json") }, () => {
  const R = JSON.parse(fs.readFileSync("outputs/reports/boyer_devilliers_full.json", "utf8"));
  for (const o of PB.selectionOptions(R)) {
    for (const ob of PB.observations(R, o)) {
      const f = Object.assign({ role: "all" }, ob.link), agg = PB.aggregate(PB.applyFilters(PB.shotsFor(R, o), f, f.view), f.view);
      const list = PB.selectShots(agg, f.zone || "");
      assert.equal(list.length, ob.count, `${o.key}: ${ob.headline}`);
      if (ob.kind !== "intercepted") assert.equal(agg.denominator, ob.n);
    }
  }
});
