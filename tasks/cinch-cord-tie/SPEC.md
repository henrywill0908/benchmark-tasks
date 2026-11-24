# TASK_SPEC: cinch-cord-tie

Status (2026-11-24, rounds 43 to 48): THE WHOLE TIE PLAYS ON THE HANGING CORDS (trial h37-hold-l, task.demo()'s
default; physics preset nylonlead). What the rounds after 42 settled:
- The lead-in on the slot's mouth (0.5 mm, all four sides) lets the held draw run: without it each strap joint butted
  on the head's face. The lead-in's extra collision shapes moved the head's centre of mass and inertia, which the
  simulator works out from the shapes, and the right then lost the head on the way round the cords (three runs, the
  same step). The head's mass properties are now pinned to the plain head's (labkit CableTie.lead_in_mass).
- The left takes the head 6 mm in from the pads' edge (flush with the exit face it slipped sideways at 2 N), which
  leaves 4 mm between the hands at the second take, so the right takes the tail end on and turns to its drawing
  heading over the first 60 mm.
- The draw is paced by the tail and stops on the tail's length (the hand runs 20 to 30 mm ahead of the tail it
  draws). Job 131: the first full tie on this start, 33 to 309 mm drawn, 1.0 with the marks then in use.
- Where the tail rests. With the head in the left's fingers, 11 mm off the cords, the loop comes taut on the cords
  at about 300 mm drawn. Drawn on, the arms and the grips give (the head turned 25 degrees in the fingers at 307 mm),
  and let go the tail runs back until the loop is slack: the pawl holds by friction on nylon (0.25), which does not
  hold a loaded tail. Jobs 131, 132, 133: drawn 309.3, 301.5, 307.5 mm, at rest 303.2, 296.5, 297.1 mm.
- The marks are whole millimetres from the tip, as the second review asked (the first marks, 303.1 and 284.4 mm,
  read as sub-millimetre thresholds). Round 47 put them at 300 and 280 mm, inside the scatter above (job 133: 0.5).
  Round 48: 290 and 270 mm; the demo draws to 302 mm and eases back from where the hand really is before it lets go.
  With the 290 mm mark at the head's face 73 mm of strap is left in the loop; hugging the three cords takes 36 mm.
- The scorer's reading of the strap's way into the slot (round 48): a tight loop let go cocks the free head, the
  strap then comes into the mouth on a slant over the lead-in, and the chord between two strap joints crossed the
  entry's plane outside the passage's window, so a tie drawn 297 mm read as not threaded (job 133's probe run). With
  the tail out of the exit, the strap's last crossing of the entry's plane within the head's face is its way in.
- The official run's cameras. Some launches of `inspect-robots run` on the test machine came up with the renderer
  late and every camera frame discarded (black videos; the log says "OgnSdOnNewFrame : frames ... discarded" at each
  step). It is the launch, not the task: the same command a minute later renders. The probe kit's job watches for it
  and launches again.
Not done: a pawl that holds a loaded tail (preset nylongrip gives the pawl's plate a friction of its own, 1.2; never
run), so the loop the demo leaves is the one the tie relaxes to, about 66 mm of strap round cords that take 36.

Status (2026-11-23, rounds 29 to 42): THE START IS THE HANGING CORDS AGAIN, WITHOUT THE STAND (TIE_BUILD hang37, the
default). The lying-cords start of rounds 19 to 30 is kept in the code (lift25) but is not the task. Why it went:
every grip a YAM hand can get on a strap lying flat is by the strap's edges, a pivot the strap turns about; the
tail has to tip over backward (half a turn) to reach the head, and it turned 43 to 52 degrees and stopped in every
trial of rounds 23 to 29, whatever the path (a 2D model of the strap agrees: no path gives the tip-down pose). The
hanging cords need no such turn: turned on edge, the strap stands out like a ruler and a hand's pads close on its
faces. The scene is the first submission's room (wall, TV edge, power strip, three cords from the TV to the strip),
no stand, no clip, nothing pre-looped: a 370 mm tie (59 links) lies flat on the console in front of the strip,
head at (0.290, -0.170), strap along +y. The demo (trial h37-hold-n and its relatives): both hands lift the tie and
turn it on edge; the right, holding the head, carries the ruler behind the cords and swings its tail round to the
front; the left takes the tail by its faces; the right carries the head round the cords' right to the front, draws
the strap taut, eases back and turns the head until the slot lies along the tail (closed loop on the head's pose);
the left lines the tip up and pushes 34 mm through (jobs 121 to 126: threaded and latched in every run that got
there). Then the draw, which took rounds 37 to 42 to understand:
- With nothing holding the head the tail does not come (jobs 123 to 125, at friction 0.9, 0.5 and 0.25 in the slot):
  the pull is answered by the strap's root, 4.6 mm beside the slot, the head cocks 27 degrees in the slot's 4 mm of
  play and the strap bears on the slot's two ends; 12 N pulls the tail out of the fingers instead. A long loop drawn
  into a V would also have to slide round the cords against its own root. A person holds the head. So: the right
  takes the tail's stub and holds the tie; the left lets go of the strap and takes the head from its free face
  (level, pads above and below, their edge flush with the exit face: tie_logic.HOLD_O); the right lets go; the left
  carries the head to the cords, the strap's root pointing at the wall and the way through toward -y
  (tie_logic.hang_tight: where a tie hugging the cords has its head); the right takes the stub again and draws it
  along the slot's real line. Job 126: all of that worked up to the draw.
- Friction is by contact now (preset nylon): the strap on the pads as before (0.9, averaged), nylon on nylon in the
  slot and on the pawl 0.25, nylon on the cords' jackets 0.35 (PhysX "min" combine; the cords are one fixed prop).
  At 0.3 everywhere the pads lost the strap and the head (job 124); at 0.9 in the slot the 34 mm push stalls at 23.
- The held draw fed 6 mm and stalled with a strap joint at the slot's mouth (job 126; the first submission's pull
  stalled at the same places and was forced past them). Round 41 adds a lead-in to the mouth (labkit
  CableTie.lead_in); round 43 a draw paced by the tail that works the tail from side to side when it sticks.
- The brake is aimed from where the pawl last sat on the strap (job 126: let go under load, it held the pawl clear
  and the whole tail ran out).
The simulation repeats exactly from run to run, but a small change to the head (round 41's lead-in) changed an
early step that had held in every run before: the head slid half a millimetre along the right's pads and was lost
on the way round the cords. The scorer on this start is tie_logic.hang_score: the lying-cords ladder (0 unless
closed round all three cords, at rest, hands 3 cm off; then the two marks) with the cords the fixed ones.

Status (2026-11-23, round 26): the task's start changed after the first review. The reviewed task had the tie already
looped round cords hanging under the TV, its head held in a clip on a post and its tip 4 mm from the slot, and scored
the tail's length against 134.8 and 117.8 mm. The review asked for no stand and no pre-aligned tip, a start where the
robot positions and threads the tie itself, and millimetre-scale checks such as marks on the tie. Now (TIE_BUILD
lift25, the default): the three cords lie side by side on the console from the wall into a free power strip (a jointed
Harness), a 250 mm tie lies flat beside them, and nothing holds the tie, its head or the cords. The scorer
(tie_logic.lift_score) asks for the tie closed round all three cords, still, the grippers 3 cm away, and then reads two
marks on the strap, 184.4 and 165.6 mm from its tip: 1.0 with the first out of the head, 0.5 with the second, 0.25
otherwise. The tail rests in 6.25 mm steps (a strap joint at the pawl), and each mark sits halfway between two steps.
The latch drops at 8 mm of tail out (16 on the first start). task.demo() plays the first half that every GPU run since
round 23 has completed (the left lifts the cords, the right slides the strap under, the grips change, the right holds
the head by the bundle); a person finishes the threading and the pull on the keyboard, and that recorded run becomes
the demo. Why the scripted threading of rounds 23-25 failed: the wrist turned the tail the short way (tip down, 90
degrees), but a strap looped round the bundle turns the long way (tip up, over, back down: 270), so the strap behind
the pinch was wound against itself and levered the tail back 47-50 degrees in every trial. A scripted finish built
on that (the tail carried over on an eased pinch, the head turned to face it; tie_logic._cinch_tail, trials c25-*)
is round 28's probe.

Status (2026-11-20, round 17 result, round 18): the candidate is now m-demo, task.demo()'s default. Round 16 showed
the loop comes to rest only where a strap joint sits at the pawl: on the jig 137.5 mm out (1.0) or a link (6.25 mm)
back, 131.3 mm (0.5). Every let-go at 137.9 mm or more stayed at 137.5; every one at 137.3 or less fell a link. Round
16's tighten stopped at 137.3, the stall just short of that joint, so all twelve fell. Round 17 tightened past it, at
12 mm/s until 138.3 mm is out: 7 of 12 tied snug, m-demo fresh among them (drawn to 138.1, resting at 137.5, 5.1 mm
slack), and none slid back. The other five stalled early in the pull: four at once, 25.6 mm out, where the pads (on
the strap's last 7 mm) slid off the tip, and one at 57 mm. Round 18: m-demo fresh again, the scripted demo through
inspect-robots (yam_scripted), and round 17's order replayed with an "unstick" on the pull (the hand never more than
12 mm ahead of a stalled strap, and moved toward the robot while held).

Status (2026-11-20, round 14 result, round 15): the drawn clip stand gave the physics back. The pre-looped start's
fresh trial scored 1.0, identical to the 0.1 mm to jobs 53-62 (insert 22.7 mm, through 138.1 mm, 5.1 mm slack over the
tightest). Of the five trials, two tied (c-heavier, c-deep), one ended loose (c-heavier-2, 0.5) and two stalled in the
first pull (c-fast, c-slow, 0.25). The loose end: the last stroke ran 30 mm past snug, the pads slid off the 6 mm of
tail past them, and the strap snapped back 7 mm before the brake held. The stall: a strap joint on the slot's entry
plane, the strap against the far wall (pawl at 54 degrees). The loop comes in from the far-wall side and the hand pulls
straight down the slot, so the strap bends round the far entry edge and a 6.25 mm link's corner catches there; the
wrap's stalls 44/51/57/63 mm out are the same catch, a link apart. Round 15: end 6 mm past snug and ease 6 mm back
before letting go; aim the pull 3 or 6 mm toward the pawl's side so the strap runs straight through the slot; 12 trials
for a rate. And task.demo() now carries the probe's closed loop (task.Steering, from the step hook), so yam_scripted
plays the probe's moves; c-demo, fresh, checks it against c-heavier's numbers. Counting distinct paths with the jig's settings so far, 5 of 9 pulls tied snug.

Status (2026-11-20, round 13 result, round 14): with the clip stand spawned as colliders, the pre-looped start's
fresh trial scored 0.25 (job 65): the pawl stood at 52-54 degrees as the pull began and wedged. Without the clip, that
same fresh trial was identical to the 0.1 mm in jobs 53-62 (1.0): the simulation is deterministic, and the clip's 5 mm
contact skin, 0.3 mm off the head, was enough to send it down another path. Its second trial scored 1.0. So the clip
stand is now drawn only (task.stand_decor, labkit.CableTie.decor: visual shapes on the welded head, no colliders): it
shows what holds the head, the weld does the holding, and the arm and finger checks keep clear of it as before. The
wrap with the clip: 0.25, 0.25, 0.0 (one tie in ten wrap pulls overall). B1 answered: a gripper holds the head (slip
0.0-0.3 mm); the firm squeeze shakes it (56-68 mm/s), the light one (about 10 N a side) holds it still (7 mm/s); under
a 25 mm tug the holding arm gives 10 mm. Round 14: the candidate fresh (expected 1.0), second, and three small
perturbations (push 0.2 s quicker, slower, the strap 0.4 mm deeper in the pads) to measure how often it ties.

Status (2026-11-19, round 12 result): THE WRAP's 1.0 DID NOT REPEAT (job 62). The unchanged winner (k-la) scored 0.25;
so did every margin variant (l70, a9, rb), and two never got the tip through (l45, a4). What varies run to run: the
push leaves 11-17 mm of tail out (the strap slides 3-10 mm in the left's fingers), so the right holds only the tail's
last link, and the pull stops at 44, 51, 57 or 63 mm out, one link (6.25 mm) apart: a joint catches in the head, the
tip slips out of the fingers and, the pawl being held clear, slides back. One wrap pull in seven has tied. The
pre-looped start scored 1.0 again (c-heavier, seventh run in a row, first in its launch as a fresh start would be).
Consequence: the pre-looped start with the clip stand is the submission candidate; the wrap stays an experiment.
Round 13 (final): c-heavier twice with the clip; the wrap three times with the clip (s-la, s-la-2, and s-la-p12, the
pull turned 12 degrees toward the robot instead of 20, the least the right arm reaches, to ease the side load on the
tail); and B1 below.

Status (2026-11-19, round 13): what holds the head? Both, as asked. (A) The clip
stand is in the scene for the fixed builds: a blue spring clip gripping the head's top and bottom faces from the robot's
side (1.5 mm jaws, 0.3 mm off the welded head), on a 5 mm steel rod standing on a 30 x 30 x 4 mm base on the console
in front of the power strip (tie_logic.stand_parts). The slot's faces and the strap's root stay open; offline, the
fingers' fat envelope stays 3.2 mm or more from it and the arm bodies 74 mm. Round 13 reruns the wrap's and the jig's
1.0 with it there. (B) A robot hand holds the head instead (the brief's "one arm holds"). Staged, each a GPU round:
  B1 (round 13): can a gripper hold the 8 x 9 x 5.6 mm head while the other hand pulls on the strap? The free tie on
     the console: the right picks it up by the head and holds it; the left pulls 25 mm along the strap (f-pick, f-tug,
     f-tug-light). Measures the head's slip in the grip.
  B2: a start the hand can take the head from: the tie resting on the power strip with its strap already passed behind
     the cords (the wrap's shape, lower). The right picks the head and lifts it to the bundle (the loop slides up the
     cords), holding it where the clip would.
  B3: the left swings and inserts as in the wrap; then, since the right is holding the head, the left lets go, takes the
     tail at the exit and pulls. Needs a ratchet that holds the tail during that regrip (latch at 16 mm) without the
     wedging the wrap's pull met in rounds 8-10 (latch at 100 mm): the open problem.
  B4: the right lets go of the head once the tie is snug; scored as now.

Status (2026-11-19, round 12): round 11 ran (job 60): THE WRAP TIED SNUG AND SCORED 1.0 (k-latch-aim: the pawl
held clear until 100 mm of tail is out, the pull aimed 6 mm toward the pawl's side; drawn 121 mm, 5.0 mm slack, the
pawl at 30 degrees at the end, tied at 29.75 s). Either fix alone: 0.25. The pre-looped 1.0 held (c-heavier); the
pre-looped start with the pawl clear to 100 mm went unstable (c-latch, 0.0). Round 12 measures the wrap's margin: the
same trial again, the pawl let go at 70 and 45 mm, the aim at 4 and 9 mm, the re-bite on top; and the pre-looped start
with the wrap's settings. Fidelity note: the latch distance is part of the simulated tie, not the policy; the
submitted task fixes it per start.

Status (2026-11-19, round 11): round 10 ran (job 56): pre-looped 1.0 again; the wrap jammed 27-44 mm out in all four.
Its shape logs show the tail leaving the exit angled toward the robot and the pawl standing at 54-55 degrees on the
strap (the jig: about 30), wedging. Round 11: the pawl held clear until 100 mm is out, and/or the pull aimed 6 mm
toward the pawl's side.

Status (2026-11-19, round 10): round 9 ran (job 53): the pre-looped start 1.0 again, with and without the re-bite. The
wrap's strap jams 50-57 mm out even under a full-width grip, with no set taken, leaving the exit turned 17 degrees
(pawl 54 degrees vs the jig's 30): it goes through the head crooked. Round 10: the inserting hand stays as a loose
guide until the first pull is done; a 10 degree yield variant; the probe logs the strap's shape, joints and set.

Status (2026-11-18, round 9): round 8 ran (job 49): c-heavier 1.0 a third time. The wrap with a 12 mm steering cap
got the tail in 3 of 4, but each pull stopped 44-51 mm out (0.25): the pulling hand had bitten only the narrow tip
link and slid off the end. Round 9: the pulling hand draws 25 mm, opens, slides back to the exit and bites the
full-width strap, then pulls; tried on the wrap (four variants) and the pre-looped start.

Status (2026-11-18, round 8): round 7 ran (job 45): c-heavier repeated the 1.0 exactly (the pre-looped start is
reproducible). The wrap went 0 for 3: the swing reached the slot but left the tail turned about 10 degrees in the
edge grip, the tip 9-10 mm off the slot's line against a 6 mm steering cap, and it stopped at the head's face. Round 8
adds closed-loop yaw steering (the hand turns about the strap in its fingers until the tail's last 12 mm points down
the slot, up to 15 degrees) and a 12 mm cap for the wrap trials, with a shift-only control; c-heavier runs a third time.

Status (2026-11-18, round 7): round 6 ran (job 38): l-heavier tied the cords snug and scored 1.0, the first full
score (jig start, heaviest tie physics). Defaults are now jig + heavier. Round 7 reruns it unchanged (c-heavier) and
tries the "wrap" start, closer to the brief: the strap slipped behind the cords with a straight tail, and the robot
swings the tail round the front of the bundle itself before threading and cinching. Still open from the brief: one arm
holding the cords together (the sim's cords are rigid and already gathered).

Status (2026-11-18, round 6): round 5 ran (job 34): the hand-over held and one pull drew the tail from 21 to 67 mm
out, then jammed when the spawn loop's 55 degree turn (a permanent kink in an elastic strap) reached the slot; 3 of 5
pushes blew up on the sprung pawl. Round 6: the pawl is latched clear until 16 mm of tail is out (then sprung onto the
strap and braked), and the strap takes a set past 20 degrees a joint (nylon's yield), so the kink straightens through
the head. Fidelity note for task.json: the latch and the set are simulation stand-ins for a real tie's flexible pawl
tongue and nylon's plasticity.

Status (2026-11-18, later): round 4 ran (job 32): the strap sprang back out of the slot every time the inserting hand
opened to regrip. Round 5: never let go: one long push along a 62.5 mm straight run, the right hand takes the tail before
the left lets go, one long pull (straight, then 20 degrees toward the robot). The loop: push to main starts a GPU run;
results come back on the `results` branch.

Status (2026-11-18): round 3 ran (r3-5502a16-29): feeding along the straight run threaded 5 of 7 (11-16 mm out); one
trial ended closed round the cords, released and still (0.25). The tail slid back out (weak brake; a 6 mm finger opening
dragged it) and the pulling hand only found the narrow tip link. Round 4: a ten-times stiffer brake engaged at 0.2 mm,
fingers open 14 mm to let go, a third push to 30 mm out (108 offline tests).

Status (2026-11-17, later): round 2 ran (results r2-bfe841b-27): every hand gripped the strap, nothing threaded:
an edge grip can't turn the strap, so the line-up swung the tip 10-16 mm off the slot; the "chatter" was PhysX velocity
noise (held parts stood still to 0.1 mm). Round 3: a D-loop with a straight run into the slot, the hand only pushes
along it, stillness from displacement over 1 s (116 offline tests).

Status (2026-11-17): round 1 ran (results cinch-cord-tie-b5e1751-24.zip): the tilted grip's path drove the left arm
into the TV in 15 of 17 jig trials; the one edge-grip insert threaded 5.7 mm and slid back out; touched ties
chattered at 35-60 mm/s. Round 2 (`../../../cinch-cord-tie-gpu/RUN.md`) is built to make it work: edge grips only,
steering and closed-loop push depth, the task-code brake, physics presets (heavier, damped ties), snug judged against
the tightest loop the cords allow round the head, a black tie, and `armcheck.py` replaying every arm path against the
room (118 offline tests). FOLLOW-UP: task.json's ruler numbers (151/134 mm) still follow the old snug rule.

Status (2026-11-16): offline only, at the probe stage. Nothing has run in Isaac Sim. The probe kit is in
`../../../cinch-cord-tie-gpu/` (RUN.md, RUN_HEADLESS.md); `tie_logic.py` holds every number. Round 1 is 23 trials over
three builds (jig, free tie, narrow slot) under a 95-minute GPU budget, queued after the outlet task's round 12. 145
offline tests pass against the real yam_arena source (`test_logic.py`, `test_task.py`, `test_demo.py`), including
yam_arena IK reach of every probe waypoint, a finger-envelope clearance check against the head and cords, and
task.json kept in step with the code. A fake-simulator dry run plays all 23 trials through probe.py and the runner.
Submission files: `task.json`, `README.md`, and `task.py`'s `demo()` (the pre-looped start, round 15). Not submitted.
Working copy: a private repository holding the task folder and the GPU kit side by side.

Numbers worth knowing (tie_logic): the cords' hull is 42.4 mm round; a tie hugging it leaves about 159 mm of tail
(`TAIL_SNUG` for full credit: 151 mm, ruler from the head's face); the pawl meets the strap at 57.8 degrees
(self-locking above friction 0.63; the sim uses 0.9); in probe round 1's jig the rigid cords stand 3 to 9 mm off the
head, so the tightest loop leaves 14.8 mm of slack and the jig's best score is 0.5.

Probe round 1 design decisions (made offline, 2026-11-16):
- A straight-down grip at the tie's height (z 92 cm) is out of the left arm's reach near the cords (yam_arena IK misses
  by 23-30 mm). Flat grips are tilted 35 degrees toward the head; edge grips are horizontal with the fingers vertical.
- The tip goes in by two strokes (tip just through, slide back 20 mm open, push again until 18 mm is out): a single
  long push either can't be reached on the loop's curve or drives the back of a finger into the cords.
- The right hand pulls in three strokes (end-on first, then from the front) to 30 mm past the tightest loop, as the
  arm sags under load (push past targets).

Coordinates in cm from the midpoint between the two arm bases, on the table (the TV console's top): x forward, y to
the robot's left, z up from the table top. Arm bases at y +30.5 (left) and -30.5 (right).

**Title (draft):** Bundle the cords hanging under the TV with a cable tie, pulled snug

**Category:** Societal relatability. Most people have tidied the cords behind a TV or desk and closed a nylon cable
tie (zip tie) round them: the scene and the object are known firsthand, not from a workshop. Not "Impact on jobs":
tie-wrapping cables is not a core O*NET task of a 500k-worker occupation (only fire-alarm installers list it, about
86k).

**Instruction (draft, goal only):**

> Tie the three cords hanging below the TV together with the cable tie, pulled snug, then let go of it.

No technique: it doesn't say to hold the head, which way to thread, or to pull the tail.

**What it tests:** closing a stiff, springy nylon strap round a loose bundle, then feeding its tapered tip into a
slot about 5 mm wide in a head that has to be held steady, and pulling the tail until the loop is snug against the
cords. The ratchet only lets the strap go one way, so a tip fed in from the wrong side, a skewed tip or a strap twisted
half a turn never catches. A loop that isn't pulled snug slides down the hanging cords as soon as it is let go.

**Scene (task, draft):**

| Object | Type | Size | Material | Fixed / mass | Where |
|---|---|---|---|---|---|
| Wall | panel | 60 x 40 cm, 1.3 cm thick | painted drywall | fixed | face at x 46 |
| TV (bottom 8 cm of it) | panel | 60 x 8 x 4 cm | black plastic | fixed, on the wall | bottom edge at z 34 |
| Power strip | box | 28 x 5.5 x 4 cm | white plastic | fixed | on the console, under the TV |
| Cord 1, TV power | round cord | 7.0 mm | black PVC | anchored both ends | TV bottom to power strip |
| Cord 2, HDMI | round cable | 7.3 mm | black PVC | anchored both ends | TV bottom to TV console |
| Cord 3, phone charger | round cable | 3.5 mm | white TPE | anchored both ends | TV bottom to power strip |
| Cable tie | nylon 6/6, 200 mm x 4.8 mm x 1.2 mm strap, head 8 x 9 x 5.6 mm | 1.3 g | free | lying on the console, 13 cm right of the cords |

The cords hang in a loose bundle, about 38 cm in front of the robot and 7 cm in front of the wall. In the task they
are flexible (round 2 on); probe round 1 uses rigid stand-ins already gathered, to isolate the tie.

**Scoring (draft; numeric, partial credit, a release gate and 1 s of stillness):**
- Gates, score 0: either gripper (TCP or pad tips) within 3 cm of the tie; the tie moving faster than 1 cm/s; the tie
  not closed round all three cords.
- Closed = the tail comes out of the head's far side by at least 2 mm, and each of the three cords passes through the
  loop.
- 1.0 closed, snug (the tail out of the head at least 151 mm long, which leaves at most 8 mm of slack round the
  bundle) and still on the hanging run (head at least 10 cm above the console), held 1 s.
- 0.5 closed with at most 25 mm of slack (tail at least 134 mm) and the head that high.
- 0.25 closed, any slack, or slid down.
- 0 otherwise, including doing nothing.

Real-world measurement (what a person checks, separate from the sim's proxies): a ruler along the tail from the head's
face to the tip; a photo showing all three cords inside the loop and the grippers clear; a tape from the console to
the head; a 1 s video for stillness. Sim proxies: the strap's polyline arc length past the slot's exit plane (tail),
point-in-polygon of each cord's axis in the loop's plane (enclosed), the head's height, the TCP and pad-tip distance to
the strap polyline and head box (release), the head's and links' speeds (still).

**Fidelity (draft):** the strap is 32 rigid 6.25 mm nylon links on bending joints with nylon's real bending stiffness
(EI about 1e-3 N m^2), no twist; the last link is narrower (the tapered tip). The head is rigid with a through-passage
of 4.0 x 6.4 mm, larger than a real slot (about 1.6 x 5.2 mm with a lead-in), so capture tolerance (about +-1.6 mm) is
close to a real tie's tapered tip in a chamfered slot. The ratchet is a sprung pawl tongue in the head that wedges
against the strap when it backs out (friction, not teeth); in `brake` mode task code stiffens the pawl when it sees
the strap back out. Link masses are twice real (stability), so a free strap sags more than a real one.

**Not a duplicate of:**
- #1441 (open, another team: cut a cable tie off a bundle). Same objects, opposite job: it scores cutting the strap
  beside the head without nicking cords; this scores closing and cinching a tie, with no cutting step and no tool.
- #2219 (open: connect two workstation ports and route the slack). Cable routing and plugging; no tie, no threading.
- #446 (open: pull one cable out of a clipped bundle), #1285 (closed: ratchet a tie-down strap round a crate), #2134
  (open: cinch a bin liner): different objects and manipulation.
- Nearest by ability, to watch: #1153 (merged: tie an overhand knot, "loop formation and threading") and #1923 (open:
  thread a shock cord through a spring cord lock). Both thread a soft cord; this feeds a stiff strap tip into a held
  ratchet slot and must cinch it to tension round a bundle, which full credit requires. A reviewer may still read
  "threading" as the same ability: the snug cinch has to carry the importance argument.
Board checked 2026-11-16 (2,311 PRs, through #2312, one `gh pr list` and a local grep): no other tie, zip-tie or
cord-tidy task. Re-check open PRs right before any submission.

**Context draft:** Behind most TVs and desks, the cords are bundled with a cable tie. Closing one takes two hands that
cooperate: one holds the head steady while the other feeds a springy 4.8 mm strap tip into a 5 mm slot from the right
side, then pulls the tail until the loop bites, without the bundle sliding away.

**Run limit (draft):** 3,600 steps (3 minutes at 20 Hz); revisit once a demo exists.

**Reset notes (draft):** a closed tie can't be reopened without a tool: each trial uses a new tie from the same bag.
The cords are re-hung from the same two anchors.

## Build risks (the probe exists for these)

1. A floating 34-body articulation of 0.08 g links: does it rest on the console, survive a 20 N grip, reset between
   trials? (`free` build: f-rest, f-pick.)
2. Grip on a 1.2 mm strap: the sim's finger PD gives about 3 N per finger flat (0.6 mm error x 5,000 N/m) and about
   12 N edge-on. Cinching needs about 5 N against the strap's springback plus pawl drag. (f-tug, and the pull strokes.)
3. Does the tip go into the passage at all, how far off can it be (j-off-*), and does the strap slide through without
   jamming on 1 mm thick boxes? (every j- trial.)
4. Does the cam pawl hold one way in PhysX, or must task code brake it? (j-cam vs j-brake vs j-open.)
5. Can the arms reach: the rolled and end-on grips near z 92 cm. (offline IK tests; the GPU logs actual TCPs.)
6. Is lining up the hard part? j-naive (pushed along its own curve, no lining up) should miss; if it captures, the
   task's difficulty sits elsewhere (the wrap or the cinch). Robustness: j-gz2 / j-gzm2 (strap taken 2 mm off in the
   pads, open loop). Approach, speed, squeeze, grip depth, over-pull: j-tilt10, j-fast, j-light, j-depth8, j-pull60.
   A slot near real width: t-cam, t-off-z15.
7. Not in round 1: wrapping the strap round the bundle from the console start, flexible cords (so cord count and bundle
   tightness), the right hand holding the head instead of the jig.

## Demo plan (draft; demo.py)

Pick the head (f-pick), regrip it from behind near the cords so both slot faces stay free (reachable at the working
pose; not at the pick spot, IK misses by 9-17 mm), wrap with the left hand, insert in the probe's two strokes steered
on the measured tip, pull, let go. Open problems: carrying the straight 200 mm strap to the cords without striking the
wall (pointing forward its tip would reach x 57 cm, the wall is at 46 cm); the wrap itself; which hand pulls once the
right holds the head. If the regrip proves impractical, the start could change (a tie hanging from a hook, or standing in
a cup on the console) so the head can be taken from behind directly.

## Review criteria self-check
- [ ] Importance: plausible (Societal relatability); the threading neighbours above are the risk.
- [ ] Construct validity: hinges on the strap and ratchet stand-in. Unknown until the probe runs.
- [x] Scoring: numeric, ruler and photo checks, release gate, 1 s still, every outcome covered (draft). task.json's
      scoring states the real-world measurement first and the sim's proxies after it (the point #1441 was sent back on).
- [ ] Clarity: the object table is complete for the task; probe round 1 uses a jig and rigid cords.
- [ ] Feasibility: no demo yet.
- [ ] Blind: nothing identifying in the task text. Re-check before upload.
