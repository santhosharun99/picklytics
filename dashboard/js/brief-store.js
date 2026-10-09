// Coaching-brief notes saved in this browser only (localStorage) - there is no account or server storage.
// Item: {id, match, who, headline, measure, scope, quality[], link (workspace URL), clips[], note, saved, auto}
(function (root) {
"use strict";
const KEY = "pb.brief.v1";
let mem = null;   // fallback when storage is blocked (private window): the brief still works for this page view
function readAll() {
  try { const raw = root.localStorage.getItem(KEY); return raw ? JSON.parse(raw) : []; }
  catch (e) { return mem || []; }
}
function writeAll(items) {
  mem = items;
  try { root.localStorage.setItem(KEY, JSON.stringify(items)); return true; } catch (e) { return false; }
}
const api = {
  available() { try { const k = "pb.t"; root.localStorage.setItem(k, "1"); root.localStorage.removeItem(k); return true; } catch (e) { return false; } },
  list(match, who) { return readAll().filter(x => (!match || x.match === match) && (!who || x.who === who)).sort((a, b) => a.saved - b.saved); },
  has(id) { return readAll().some(x => x.id === id); },
  add(item) { const all = readAll().filter(x => x.id !== item.id); all.push(Object.assign({ note: "", saved: Date.now() }, item)); return writeAll(all); },
  remove(id) { return writeAll(readAll().filter(x => x.id !== id)); },
  setNote(id, note) { const all = readAll(); const x = all.find(y => y.id === id); if (x) { x.note = note; x.edited = Date.now(); } return writeAll(all); },
};
root.PBBrief = api;
})(window);
