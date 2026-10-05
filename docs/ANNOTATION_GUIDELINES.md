# Annotation schema and guidelines (pickleball video)

Tools: `dashboard/label.html` (rally labeller: events on the video timeline) and `dashboard/review.html`
(shot-type review queue). Labels are saved as JSON in `labels/`.

## 1. Unit of annotation

| Level | What it is | Fields |
|---|---|---|
| **Video** | one match video | `video`, `fps`, players `A` / `B` (names) |
| **Rally** | one point, first serve contact to the end of play | `id`, `start` / `end` frame, `winner` (A / B), `ending`, `reviewed` |
| **Event** | one moment inside a rally | `kind` (hit / bounce), `frame`, attributes below |

## 2. Events

### Hit
- **When:** the frame where the **paddle touches the ball**. One or two frames off is acceptable
  (the evaluation matches within ±4 frames).
- **Who:** `side` = **near** (player at the bottom of the screen, key `1`) or **far** (top, key `2`); `player` = A / B.
- **Attributes:**

| Attribute | Values | Rule |
|---|---|---|
| `shot` | serve, return, drive, drop, dink, reset, lob, overhead | see definitions below; serve and return are filled in automatically |
| `stroke` | forehand, backhand | the side of the body the ball is hit on (for the hitter's dominant hand) |
| `volley` | true / false | hit **before** the ball bounced - independent of shot type (a drive can be a volley) |
| `xy` | pixel position | optional: click on the ball at contact |

### Bounce
- **When:** the frame the ball **touches the court**. Key `X`.
- `out` = true (key `U`) if it landed outside the court lines.
- Every rally has at least two (serve and return must bounce). A ball dropping into the net does not need one.

## 3. Shot-type definitions (current taxonomy)

| Shot | Definition | Decision rule |
|---|---|---|
| **Serve** | the first shot of the rally | automatic |
| **Return** | the receiver's first shot | automatic |
| **Drive** | any hard or medium-pace ball hit after the bounce or in the air, flat-ish over the net | "hit hard or medium" - pace comes from measured ball speed |
| **Drop** | soft ball that falls into the opponent's kitchen, hit from the **back or middle** of the court | hitter deep + soft + lands in / near the kitchen |
| **Dink** | soft ball into the kitchen hit from the **kitchen line** | hitter at the net + soft + lands in the kitchen |
| **Reset** | soft block of a **fast incoming ball** that takes the pace off (defensive) | incoming ball fast + soft reply |
| **Lob** | high arcing ball that lands **deep**, over or behind the opponent | high arc + deep landing; high arc landing in the kitchen is a **drop** |
| **Overhead** | smash hit from above the head | contact above head height |

Order to decide: hard → **drive**; soft into the kitchen → **drop** (from deep) or **dink** (from the net);
soft block of a fast ball → **reset**; high and deep → **lob**; from above the head → **overhead**.
Third shots are labelled by type only (the "third-shot" prefix is derived automatically).

## 4. Review-queue answers (quality options)

| Answer | Key | Use when | Used for training? |
|---|---|---|---|
| a shot type | D P K T L O | clear | yes |
| **Unclear** | U | can't tell | no |
| **Not a hit** | N | nothing was hit: missed swing, ball into the net, ball picked up after the rally, a bounce | no (kept as hit-detector negatives) |
| **Wrong player / marker** | W | a real shot, but the dashed box is on the wrong player or the ring is not on the ball | no |

Rally `ending`: winner, forced error, unforced error, fault, unclear.

## 5. Rules that keep the labels consistent
1. **Unclear beats guessing** - use U freely.
2. **Model guess hidden** while labelling (no anchoring); auto-labelled rallies count only after review (`Enter`).
3. **Merge classes people cannot separate reliably** - "groundstroke" was merged into "drive" (2026-10-04).
4. **Look at who is at the ring**, not who the box says, in doubles.
5. **Note the context**: pre-match practice clips are labelled the same way but recorded as practice.

## 6. How the labels are used
- **Training:** hit / bounce deciders, the shot-type model.
- **Ground truth:** accuracy shown on the site (hits 86-94%, forehand / backhand 95%, shot types per class on matches
  the model never saw).
- **Selection (active learning):** the review queue shows 40% least-certain predictions, 35% likely rare / soft shots,
  25% random.

## 7. Change log
- 2026-10-04: shot-type mode added (types hidden on import); "Not a hit" and "Wrong player / marker" answers;
  groundstroke merged into drive; smash = overhead.
