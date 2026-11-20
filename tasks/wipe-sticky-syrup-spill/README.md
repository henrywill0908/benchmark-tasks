# Clean a dried syrup spill off a table with a spray bottle and a towel

Instruction given to the policy: `Clean the dried syrup off the table.`

The instruction leaves the procedure out. Dried syrup smears under a dry cloth, so the work is:
1. Take the trigger spray bottle and work its trigger with the hand while holding it tipped toward the spill: a light
   grip carries it, a hard squeeze sprays.
2. Wet the whole spill, then stand the bottle back up.
3. Press the folded towel onto the table and wipe until no syrup is left and nothing has been smeared beyond the spill.

Wiping the dried syrup without wetting it first drags it along in streaks. Touching the syrup with bare fingers
scores 0.

Files:
- `task.json`.
- `wipe_sticky_syrup_spill.py`: the scene, `score()`, `success()` and `demo()`.
- `syrup_logic.py`: every number and rule (the syrup field, the spray, the wipe, the score) and the plans the demo was
  recorded from.
- `labkit.py`: object kinds the stock runtime lacks: the trigger spray bottle, true cloth for the towel, and the colour
  layer the syrup is drawn with.
- `kit_replay.py` and `demo_actions.csv`: the recorded demo and its player.
- `hulls.py` and the offline checks: `test_logic.py`, `test_task.py`, `dryrun.py` and `armcheck.py`.

No third-party assets: every mesh (the bottle, the syrup layer, the towel) is generated in code by the files above.

## Environment

The program's Brev launchable (Isaac Lab-Arena with Isaac Sim, L40S or similar) with yam-arena and inspect-robots as
it installs them. Before every shell: `source ~/IsaacLab-Arena/yam.env`; the Dream key is in `~/.dreamlabs/credentials`.

## Offline checks, no GPU

From this folder:

```bash
python -m pytest -q -p no:cacheprovider test_logic.py test_task.py   # 59 passed
python dryrun.py          # each scripted variant through the scorer in a kinematic stand-in
python armcheck.py        # the demo's poses through yam_arena's IK: worst error per arm (under 0.1 mm)
```

## Run and score one Dream-1 trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/wipe-sticky-syrup-spill/wipe_sticky_syrup_spill.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --instruction "Clean the dried syrup off the table." \
  --scorer task_score --max-steps 2400 --seed 0 --no-prompt --store-frames --log-dir ~/yam-eval/syrup-dream
```

`task_score` is 1.0 when `success()` has held for 1 s; otherwise it is `score()` at the last step. `SYRUP_DEBUG=1`
prints the task code's state twice a second: sprays, clean and dirty squares, syrup left, the towel's load and
water, and the let-go checks.

## Run and score the full-score demo

Same command with `--policy yam_scripted`. `demo()` replays `demo_actions.csv`: the 14 joint targets the runner
executed at each control step of a full-score trial, played row by row by `kit_replay.py`. Every row passes the
runner's guards unchanged, and the file is pinned by its sha256. The replay is open loop. With `object_jitter=0`
this recording scored 1.0 each of the three times it was replayed: once in the probe that recorded it, and twice with
the standard command, at 320x176 and at 1280x704. A recording from an earlier version replayed at full score too.

The recorded trial takes 61 s of the 120 s limit:
1. The right arm comes down over the sprayer's head with its fingers open, closes them lightly on the trigger and the
   back of the head, and lifts the bottle.
2. It tips the bottle 55 degrees and sprays the spill in a 3 x 3 pattern of nine squeezes, letting the trigger out
   between them, then stands the bottle back on its mark and lets go.
3. The left arm, its fingers closed and leaning 30 degrees, presses the towel onto the table and drags it to the
   spill. It wipes in back-and-forth strokes. When its fingertips have crept 4 cm from the towel's middle (the cloth
   follows them only about 80% of the way), it lifts and presses again on the towel's middle. When the strokes are
   done, it wipes again any square that still shows syrup.
4. It lifts off, leaving the towel on the table.

The recording comes from a scripted controller that read the towel's position and the squares' state each step. The
accepted din-rail-mount demo reads object states the same way.

Other scripted runs, all on the GPU (`syrup_logic.VARIANTS`; set `DEMO_VARIANT` in the environment to play one as an
open-loop plan):

| Run | What it does | Score |
|---|---|---|
| zigzag, circles, edges inward | the same spray, other wiping motions | 1.0 (71 s, 101 s, 104 s) |
| `dry_wipe` | no spray: wipe with the dry towel | 0 (30 squares smeared, none clean) |
| `spray_only` | spray, no wipe | 0 |
| `bare_fingers` | spray, then wipe with bare fingertips | 0 (fouled) |
| `poke_then_wipe` | spray, touch the syrup with bare fingertips, then wipe it all clean | 0 (fouled) |
| `upright_squeeze` | squeeze the trigger with the bottle standing upright, then wipe | 0 (the spray misses the table) |

## Simulation fidelity

PhysX has no liquids and no adhesion, so the syrup is task code. The spill is a 26 x 26 field of 1 cm cells, each
holding syrup (1.0 is the dried puddle's middle, 0.7 at its rim) and water. It is drawn as a layer of 1 cm tiles,
dark amber where thick, paler where thin or wet; the layer has no collision.

- **Spray.** A trigger squeeze past 8.5 mm (of 10 mm travel, against a 3 N/mm spring) sprays once; the trigger must
  come back past 5 mm before it sprays again. The spray wets a patch where the nozzle's line meets the table (about
  5 cm across at 15 cm range), most water in its middle. Where the towel lies in that patch, the towel takes the
  water instead. A bottle tipped past level sprays nothing: the dip tube draws air.
- **Softening.** Water softens syrup in proportion to the water per unit of syrup; 0.5 of each softens it fully.
- **Wipe.** A fingertip counts as pressing when it is within 5.5 mm of the table with towel cloth under it. When it
  moves, the cloth under it moves too, and the smaller of the two movements is the rub.
  - Over each 1 cm cell the pressed cloth crosses, the towel picks up syrup: little from dried syrup (1/m), a lot
    from softened syrup (21/m), so about 15 cm of rubbing clears a soft cell.
  - It also drags syrup onto the next cell along the wipe: a lot when the syrup is dried (25/m, which is the smear),
    little when it is softened (4/m). Water moves with the syrup it thins.
  - The towel keeps what it picks up. The fuller it gets, the less it takes and the more film it leaves where it
    passes.
  - It soaks up water as it wipes, and a damp towel wets what it rubs, so a towel sprayed directly also cleans,
    slowly.
- **The objects.** The towel is PhysX cloth (a surface deformable). The bottle and its trigger are rigid bodies on a
  sprung slide joint. No fluid is simulated.

## Videos and screenshot

```bash
for log in ~/yam-eval/syrup-*/*.json; do inspect-robots video "$log"; done
```

## Run limit, runtime and seeds

- Run limit: 2,400 steps (120 seconds at 20 Hz).
- One fixed layout (`-E object_jitter=0`). Dream-1 is hosted and not seeded; the demo replay is deterministic in
  practice (three replays of this recording, three full scores).
- Wall clock per trial on an RTX 4090: about 6 minutes for the demo (1,288 steps at 320x176) and about 14 minutes
  for a full 2,400-step Dream-1 run.

## Protecting the robot

Water and syrup must not reach the robot. Put disposable nitrile finger cots over the four finger pads, and change
them every trial. The spray points down at the table, away from the arms, and only the towel touches the syrup: a
bare fingertip on it scores 0 anyway.

## Reset between trials

1. Wash the table with warm water, dry it, and lay a card ring of 10 cm inside diameter on the mark centred at
   (37, 12).
2. Pour 10 g of pure maple syrup (Grade A, dark colour) into the ring and spread it to the ring's edge with a spatula. Lift the ring off, and
   leave the syrup 10 minutes at 20 to 24 C so that it skins over and turns tacky.
3. Refill the bottle to 200 mL with warm water and stand it on its mark, nozzle away from the robot.
4. Lay a dry towel, folded once, flat on its mark.
5. Put fresh finger cots on the pads.
