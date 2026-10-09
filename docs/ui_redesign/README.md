# Coach workspace redesign (2026-10-09)

`before/` = the site before the redesign, `after/` = the same pages after (real report data, rendered with
headless Chromium; `after/clip_playing_real_browser.jpg` is from the app's browser pane, which has the H.264 codec).
`brief_print_example.pdf` = the coaching brief printed to A4.

What changed, and why, is in PROGRESS.md (2026-10-09 entry). Code: `dashboard/hub.html` + `dashboard/js/workspace.js`
(match workspace), `dashboard/js/coach-core.js` (shared counting / state logic, tested in `tests/js/`),
`dashboard/js/brief-store.js` (brief saved in the browser), `dashboard/scout.html` (coaching brief),
`dashboard/player.html` (profile), `dashboard/theme.css` (shared look). `match.html` now forwards to the workspace.
