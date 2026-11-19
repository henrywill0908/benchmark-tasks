# Key a pulley onto a shaft and lock it with its set screw

Instruction given to the policy: `Fit the pulley to the shaft and lock it with its set screw.`

The instruction leaves the procedure out. The scored steps: pick the key off its mat, set it in the keyseat and press it
fully home (about 10 N); turn the pulley until its keyway lines up with the key (within 4 deg) and slide it over the key
to the shoulder; take the T-handle driver from its cup, put its bit in the set screw's socket and turn it in (about one
turn, two strokes of the wrist) until the driver's 2.0 N m clutch slips; take the driver out and put it down. A pulley
forced over a key that is not home cocks the key and scores 0.

Files: `task.json`; `task.py` (scene, `score()`, `success()`, the scripted `demo()`); `pulley_logic.py` (every number
and rule: the key's fit, the hub's gate over the key, the screw and clutch, the score, the demo plan); `labkit.py`
(object kinds the stock runtime lacks: the pulley on its shaft, loose tools); `test_logic.py`, `test_task.py`,
`dryrun.py` and `armcheck.py` (offline checks).

## Environment

The program's Brev launchable (Isaac Lab-Arena with Isaac Sim, L40S or similar) with yam-arena and inspect-robots as it
installs them. Before every shell: `source ~/IsaacLab-Arena/yam.env`; the Dream key is in `~/.dreamlabs/credentials`.

## Offline checks, no GPU

From this folder:

```bash
python -m pytest -q -p no:cacheprovider test_logic.py test_task.py   # 37 passed
python dryrun.py         # each scripted variant through the scorer in a kinematic stand-in
python armcheck.py full  # the demo through yam_arena's IK: worst error per arm (under 0.1 mm)
```

## Run and score one Dream-1 trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/key-pulley-set-screw/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the pulley to the shaft and lock it with its set screw." \
  --scorer task_score --max-steps 3000 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/pulley-dream
```

`task_score` is 1.0 when `success()` has held for 1 s, otherwise `score()` at the last step.

## Run and score the full-score demo

Same command with `--policy yam_scripted`. Open loop and deterministic, about 97 s of the 150 s limit. The right arm
picks the key, sets it in the keyseat, lets go and presses it home with closed fingertips. The left arm takes the
pulley's rim (fingers across its width) and turns it 65 deg onto the key's line, pushes the rim's face with closed
fingers until the pulley meets the shoulder, lifts the driver out of its cup, puts the bit in the screw, turns it
clockwise in two 180 deg strokes (letting go and turning the wrist back between them) until the clutch slips, pulls
the bit out, puts the driver back in its cup, and both arms go back to ready. `PULLEY_DEBUG=1` prints the task code's
state (key, slide, phase, screw turn, clutch, score) every half second.

Degenerate and tolerance runs (same command, `DEMO_VARIANT` in the environment):

| `DEMO_VARIANT` | What it does | Score |
|---|---|---|
| `no_key` | no key; the pulley turned, pushed to the shoulder and the screw turned on bare shaft | 0 |
| `proud` | the key set in its seat but not pressed home; the pulley pushed on | 0 (key cocked) |
| `no_turn` | the keyway left 65 deg off the key; the pulley stops at the key's end | 0.25 |
| `leave_driver` | everything, but the driver left in the screw | 0.75 |
| `turn_err3` | the pulley turned 3 deg short of the key's line | 1.0 |
| `turn_err6` | the pulley turned 6 deg short | 0.25 |

## Videos and screenshot

```bash
for log in ~/yam-eval/pulley-*/*.json; do inspect-robots video "$log"; done
```

## Run limit, runtime and seeds

- Run limit: 3,000 steps (150 seconds at 20 Hz).
- One fixed layout (`-E object_jitter=0`). Dream-1 is hosted and not seeded; the demo is deterministic.
- Wall clock per trial: about 9 to 10 minutes at 320x176 on an RTX 4090 (the scripted demo, 1,950 steps).

## Reset between trials

Back the set screw out 300 deg from tight (its tip then 0.3 mm above the keyway's roof), slide the pulley back until
its rear face is at the start mark (x 26.2) with its keyway turned 65 deg toward +y, lift the key out and lay it on the
mat, and stand the driver in its cup.

## Simulation fidelity

Rigid bodies. PhysX can't model a 6 N9 interference fit, a thread or a clutch, so task code does (stated in
pulley_logic.py): the simulated keyseat is 6.4 mm wide for a 6.0 mm key, 1.5 mm longer at its free end than drawn and
its top edges broken 0.6 mm (the arm places a key to about a millimetre; a key caught on a square lip sits perched),
and a hidden pad in it holds a key 1.5 mm proud until it is pressed down with 10 N, then lets it reach the floor,
where a key that is home is fixed in place. The pulley rides two joints on the shaft (slide and turn, lightly damped,
no bore contact); its keyway can pass the key's end only with the keyway within 4 deg of the key, the key home and the
screw clear of the keyway (else the pulley stops at the key; pushed on with about 20 N while lined up over a proud
key, the key cocks). The set screw advances 0.8 mm a turn while the driver's bit is in its socket (the bit is held on
the screw's axis once it drops in within 2 mm and 10 deg (a bit pushed onto the screw's face off the socket rests
there until it lines up), and comes out when the hand holding the handle lifts it 5 mm); the clutch slips 30 deg past
the screw touching the key (estimated angle to 2.0 N m), and a tight screw fixes the pulley where it is. Masses: key
20 g (a real one is 7 g), pulley 150 g, driver 60 g. The free end has a stop (a real pulley could slide off).

## Assets

No third-party assets: every shape (shaft, keyseat, pulley, key, driver, cup) is generated in code (`task.py`,
`labkit.py`).
