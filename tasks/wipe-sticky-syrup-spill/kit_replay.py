"""kit_replay: play a recorded run of a task as its scripted demo, through the official inspect-robots runner.

A recording is the 14 joint targets the runner executed at each control step of one trial, in the yam_arena
embodiment's order (left_j0..left_j5, left_gripper, right_j0..right_j5, right_gripper; joints in radians, grippers
0 closed to 1 open). A task folder carries it as demo_actions.csv next to this file, and the task file plays it
with the standard command (inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=...):

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))  # kit_replay.py sits next to this file
    import kit_replay  # noqa: E402

    DEMO_SHA256 = "..."  # python kit_replay.py check demo_actions.csv prints it


    def demo():
        return kit_replay.replay_demo("demo_actions.csv", expect_sha256=DEMO_SHA256)

The task file's own sha256 (the run's environment_revision) then covers the recording too. DEMO_SHA256 is the
file's sha256 with CRLF line ends read as LF (:func:`demo_sha256`), so a checkout that changes the line ends still
matches; a value pinned by the older rule (the raw bytes' sha256) is accepted for the same file.

How it plays. yam_scripted turns a demo's waypoints into joint targets with IK, which can't reproduce recorded
joints exactly. Importing this module patches yam_scripted (at once if it is already imported, else when it is):
when a task's demo() returns a recording, the policy sends row k of the file at control step k, then holds the
last row for hold_s seconds (-P hold_s, 3 by default, as yam_scripted holds after a plan) and asks the runner to
stop at step rows - 1 + round(hold_s * control_hz). Any other demo plays exactly as before. Nothing in the
simulator or the runner is edited.

Why the rows reach the robot unchanged. The runner's default guards clamp each action to the joint limits, then
limit each value to 5% of its joint's range from the previous executed action (the first action has no step
limit). check_actions() refuses a file holding any row they would change, using the guards' own float64
comparisons, so the actions executed in a replay are the file's rows bit for bit. Rows copied from a trial's
action log pass by construction: they are what the guards approved in that trial.

Command line (standard library and NumPy only):

    python kit_replay.py check demo_actions.csv            # rows, limits, sha256 and the task-file lines
    python kit_replay.py pack <log dir> --trial N --task <task folder> [--media <folder>] [--scene-changed]
    python kit_replay.py verify <log dir> <task folder>/demo_actions.csv [--run <run id>]

pack refuses a task file that imports assetlib (official runs can't import it: freeze the scene into a new task
folder, record on that folder and pack it), and checks the scene stamp keyboard_run.py puts on every trial
(trial_metadata["kit_scene"]): a trial
recorded on a library scene, or on scene files (labkit.py, helper modules, assets/) that differ from the task folder's
now, is refused unless --scene-changed is given (then verify must PASS).

Open loop: the replay sends the recorded targets whatever happens, so it reaches the recorded score only when the
simulator repeats the recorded run (the same start with object_jitter=0, the same physics). Contact-rich or cloth
motion can diverge. verify says whether a replay scored full marks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.abc
import importlib.util
import io
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "robocurve-kit-recording/1"
JOINT_LOG_FORMAT = "robocurve-kit-joint-log/1"
ACTION_DIM = 14
COLUMNS = (
    *(f"left_j{i}" for i in range(6)),
    "left_gripper",
    *(f"right_j{i}" for i in range(6)),
    "right_gripper",
)

# The runner's guard limits for yam_arena, as a real run's record.json reports them (inspect-robots 1.0.0): the
# clamp box is the URDF's joint limits per arm plus the gripper's 0..1, and the per-step limit is the runner's
# default, 5% of each range (no max_step is declared). runner_limits() derives the same numbers from a live action
# space when inspect-robots is there.
_ARM_LOW = (-2.61799, -8.88178e-16, 0.0, -1.69297, -1.5708, -2.0944, 0.0)
_ARM_HIGH = (3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944, 1.0)
RUNNER_LOW = np.array(_ARM_LOW + _ARM_LOW, dtype=np.float64)
RUNNER_HIGH = np.array(_ARM_HIGH + _ARM_HIGH, dtype=np.float64)
RUNNER_MAX_DELTA = 0.05 * (RUNNER_HIGH - RUNNER_LOW)
GUARD_NAMES = ("clamp", "delta-limit")

READY = (0.0, 1.0, 1.0, -0.6, 0.0, 0.0)  # yam_arena's reset pose for each arm's joints (informational only)
DEFAULT_HZ = 20.0
DEFAULT_HOLD_S = 3.0  # yam_scripted's default hold_s
DEFAULT_RUN_LIMIT = 2400
DEMO_FILE = "demo_actions.csv"
TOP_CAMERA = "top_cam"
MAX_VIDEO_BYTES = 50_000_000  # the program's limit for a video, "up to 50 MB" (read as decimal megabytes)
MAX_SCREENSHOT_BYTES = 10_000_000  # and for the screenshot, "up to 10 MB"
FULL_MARKS = 1.0 - 1e-9  # a partial-credit sum such as ten 0.1s can land a hair under 1.0

MARKER = "_kit_replay_demo"  # set on Replay objects; any copy of this module recognises any other copy's
PATCH_ATTR = "_kit_replay_patch"  # on the patched class: the methods it wrapped
STATE_ATTR = "_kit_replay_state"  # on a policy instance during a replayed trial

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FRAME_OK = b"abcdefghijklmnopqrstuvwxyz0123456789._-"
_TRIAL_RE = re.compile(r"^(?P<scene>.+)-e(?P<epoch>\d+)$")
# An absolute path: a home folder, macOS's private and per-user temporary folders, the system's temporary folder, a
# mounted volume or a Windows drive letter (written with one backslash, or two in JSON). The same pattern as the kit's
# tools/common.py ABSOLUTE_PATH (this file ships alone, so it keeps its own copy; the kit's tests keep the two the
# same). No example is written out here: this file is checked with it before pack copies it into a task folder.
_PATH_RE = re.compile(r"/(?:Users|home|root|private|tmp|Volumes|var/folders)/|(?<![\\\w])[A-Za-z]:\\")


class PackError(ValueError):
    """pack or verify refused: the message says why and what to do."""


def _num(x: Any) -> str:
    """A number exactly, as Python writes a float (not numpy's np.float64(...))."""
    return repr(float(x))


# -- files ------------------------------------------------------------------------------------------------------------


def file_sha256(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256(data: bytes) -> str:
    """sha256 of text bytes with CRLF line ends read as LF: the same value after a checkout that changes them."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def demo_sha256(data: bytes) -> str:
    """A recording CSV's DEMO_SHA256: :func:`lf_sha256` of its bytes (csv_bytes writes LF after the '#' line and
    CRLF after each row, as Python's csv module does; git may store or check out either)."""
    return lf_sha256(data)


def csv_bytes(rows: Any, control_hz: float) -> bytes:
    """The kit's recording CSV for these rows, byte for byte what recording.Recording.to_csv writes: a '#' line
    with the format, the rate and the row count, the 14 column names, then one row per control step with every
    value at 17 significant digits (so it reads back bit for bit)."""
    a = np.asarray(rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM:
        raise ValueError(f"rows must be (N, {ACTION_DIM}), got shape {a.shape}")
    buf = io.StringIO(newline="")
    buf.write(f"# {FORMAT} control_hz={float(control_hz):g} steps={len(a)}\n")
    writer = csv.writer(buf)
    writer.writerow(COLUMNS)
    for row in a:
        writer.writerow([f"{v:.17g}" for v in row])
    return buf.getvalue().encode("utf-8")


def _parse_rate(value: Any, where: str) -> float:
    try:
        hz = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: control_hz must be a number, got {value!r}") from None
    if not (math.isfinite(hz) and hz > 0):
        raise ValueError(f"{where}: control_hz must be a positive number, got {value!r}")
    return hz


def _record_of(actions_path: Path) -> Path:
    """<log>/actions/<run>/<trial>.jsonl -> <log>/records/<run>/<trial>/record.json (eval.py's layout)."""
    run_dir = actions_path.parent
    return run_dir.parent.parent / "records" / run_dir.name / actions_path.stem / "record.json"


def _load_csv(path: Path, text: str) -> tuple[np.ndarray, dict]:
    tokens: dict[str, str] = {}
    has_format = False
    header_seen = False
    rows: list[list[float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            for part in line[1:].split():
                if part == FORMAT:
                    has_format = True
                elif "=" in part:
                    key, _, value = part.partition("=")
                    tokens[key] = value
            continue
        where = f"{path.name} line {lineno}"
        if not header_seen:
            names = tuple(f.strip() for f in line.split(","))
            if names != COLUMNS:
                raise ValueError(f"{where}: the first row must name the 14 columns {','.join(COLUMNS)}; got {line!r}")
            header_seen = True
            continue
        fields = line.split(",")
        if len(fields) != ACTION_DIM:
            raise ValueError(f"{where} (row {len(rows)}): {len(fields)} values, expected {ACTION_DIM}")
        try:
            row = [float(f) for f in fields]
        except ValueError as exc:
            raise ValueError(f"{where} (row {len(rows)}): {exc}") from None
        bad = [COLUMNS[j] for j, v in enumerate(row) if not math.isfinite(v)]
        if bad:
            raise ValueError(f"{where} (row {len(rows)}): not a finite number in {bad}; the runner refuses such a row")
        rows.append(row)
    if not has_format:
        raise ValueError(f"{path.name}: not a {FORMAT} file (its '#' line must name the format)")
    if "control_hz" not in tokens:
        raise ValueError(f"{path.name}: its '#' line gives no control_hz=")
    if "steps" not in tokens:
        raise ValueError(f"{path.name}: its '#' line gives no steps=")
    if not header_seen:
        raise ValueError(f"{path.name}: no column row ({','.join(COLUMNS)})")
    try:
        steps = int(tokens["steps"])
    except ValueError:
        raise ValueError(f"{path.name}: steps={tokens['steps']!r} is not a whole number") from None
    if steps != len(rows):
        raise ValueError(f"{path.name}: the '#' line says steps={steps} but the file has {len(rows)} rows (cut short?)")
    hz = _parse_rate(tokens["control_hz"], path.name)
    return np.array(rows, dtype=np.float64).reshape(-1, ACTION_DIM), {"control_hz": hz, "rate_from": "file"}


def _load_jsonl(path: Path, text: str) -> tuple[np.ndarray, dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path.name}: empty")
    try:
        header = json.loads(lines[0])
    except ValueError as exc:
        raise ValueError(f"{path.name} line 1: not JSON ({exc})") from None
    if not isinstance(header, dict) or header.get("kind") != "header":
        raise ValueError(f"{path.name} line 1: not an inspect-robots action log header")
    if header.get("action_dim") != ACTION_DIM:
        raise ValueError(f"{path.name}: action_dim {header.get('action_dim')!r}, expected {ACTION_DIM}")
    labels = header.get("labels")
    if labels is not None and tuple(labels) != COLUMNS:
        raise ValueError(f"{path.name}: its labels {labels} are not the yam_arena order {list(COLUMNS)}")
    rows: list[list[float]] = []
    for lineno, raw in enumerate(lines[1:], start=2):
        try:
            entry = json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{path.name} line {lineno}: not JSON ({exc})") from None
        t, action = entry.get("t"), entry.get("action")
        if t != len(rows):
            raise ValueError(f"{path.name} line {lineno}: t={t!r}, expected {len(rows)} (steps must run 0, 1, 2, ...)")
        if not isinstance(action, list) or len(action) != ACTION_DIM:
            raise ValueError(f"{path.name} line {lineno}: the action must be {ACTION_DIM} numbers")
        row = [float(v) for v in action]
        if not all(math.isfinite(v) for v in row):
            raise ValueError(f"{path.name} line {lineno}: not a finite number in the action")
        rows.append(row)
    meta = {k: header.get(k) for k in ("run_id", "scene_id", "epoch")}
    record = _record_of(path)
    if record.is_file():
        try:
            conf = json.loads(record.read_text(encoding="utf-8")).get("configuration") or {}
            meta["record_hz"] = (conf.get("embodiment") or {}).get("control_hz")
        except (OSError, ValueError, AttributeError):
            meta["record_hz"] = None
    return np.array(rows, dtype=np.float64).reshape(-1, ACTION_DIM), meta


def load_actions(path: str | os.PathLike, *, control_hz: float | None = None) -> tuple[np.ndarray, dict]:
    """(rows as float64 (N, 14), meta) from a kit recording CSV or an inspect-robots action log (.jsonl).

    The CSV's '#' line gives the rate. An action log has none, so its rate is ``control_hz`` or the trial's
    record.json (configuration.embodiment.control_hz); when both are known they must agree."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"no file at {path}")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{path.name}: not a text file") from None
    if path.suffix.lower() == ".csv":
        rows, meta = _load_csv(path, text)
        if control_hz is not None and not math.isclose(_parse_rate(control_hz, "control_hz"), meta["control_hz"]):
            raise ValueError(f"{path.name} is at {meta['control_hz']:g} Hz, not {float(control_hz):g} Hz")
        source = "csv"
    elif path.suffix.lower() == ".jsonl":
        rows, meta = _load_jsonl(path, text)
        found = meta.pop("record_hz", None)
        given = None if control_hz is None else _parse_rate(control_hz, "control_hz")
        if found is not None:
            found = _parse_rate(found, f"{_record_of(path).name} next to {path.name}")
        if given is not None and found is not None and not math.isclose(given, found):
            raise ValueError(f"{path.name}: its run was at {found:g} Hz (record.json), not {given:g} Hz")
        if given is None and found is None:
            raise ValueError(f"{path.name}: an action log carries no rate and no record.json was found next to its "
                             "run; pass control_hz (--control-hz)")
        meta["control_hz"] = given if given is not None else found
        meta["rate_from"] = "argument" if given is not None else "record.json"
        source = "jsonl"
    else:
        raise ValueError(f"{path.name}: expected a .csv recording or an action log .jsonl")
    if len(rows) == 0:
        raise ValueError(f"{path.name}: no rows")
    raw = hashlib.sha256(data).hexdigest()
    # a CSV's sha256 is its DEMO_SHA256 (line ends read as LF); an action log's is its bytes' (never pinned)
    meta.update(source=source, path=str(path), sha256=demo_sha256(data) if source == "csv" else raw,
                raw_sha256=raw, steps=len(rows))
    return rows, meta


# -- the runner's limits ----------------------------------------------------------------------------------------------


def runner_limits(action_space: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, high, max_delta) the runner's default guards use: from an inspect-robots Box with the guard's own
    rule (a declared semantics.max_step entry wins, else 5% of the range), or the recorded yam_arena values."""
    if action_space is None:
        return RUNNER_LOW.copy(), RUNNER_HIGH.copy(), RUNNER_MAX_DELTA.copy()
    if action_space.low is None or action_space.high is None:
        raise ValueError("the action space declares no low/high bounds")
    low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
    delta = 0.05 * (high - low)  # DeltaLimitGuard's default for every dimension without a declared max_step
    declared = getattr(getattr(action_space, "semantics", None), "max_step", None)
    for i, entry in enumerate(declared or ()):
        if entry is not None:
            delta[i] = float(entry)
    return low, high, delta


def check_actions(rows: Any, control_hz: float = DEFAULT_HZ, *, low: Any = None, high: Any = None,
                  max_delta: Any = None, tol: float = 0.0) -> dict:
    """Refuse rows the runner's default guards would change (ValueError naming the row, which is the control step,
    its time, the column and the numbers); return a summary of rows that pass.

    Clamp: every value inside [low, high]. Step limit: for each row k >= 1, every value inside
    [row(k-1) - max_delta, row(k-1) + max_delta], computed in float64 as the guard computes it, so a value exactly
    on a limit passes and one a single float step past it fails. Row 0 has no step limit (the guard has no
    previous action yet). ``tol`` widens both checks; the guards have none, so keep it 0 for a replay that must
    be exact."""
    try:
        hz = float(control_hz)
    except (TypeError, ValueError):
        hz = math.nan
    if not (math.isfinite(hz) and hz > 0):
        raise ValueError(f"control_hz must be a positive number, got {control_hz!r}")
    a = np.asarray(rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != ACTION_DIM:
        raise ValueError(f"rows must be (N, {ACTION_DIM}): got shape {a.shape}")
    if len(a) == 0:
        raise ValueError("no rows")
    d_low, d_high, d_delta = runner_limits()
    low = d_low if low is None else np.asarray(low, dtype=np.float64).reshape(-1)
    high = d_high if high is None else np.asarray(high, dtype=np.float64).reshape(-1)
    delta = d_delta if max_delta is None else np.asarray(max_delta, dtype=np.float64).reshape(-1)
    tol = float(tol)

    def at(k: int, j: int) -> str:
        return f"row {k} (control step {k}, {k / hz:.2f} s): {COLUMNS[j]} (column {j})"

    bad = ~np.isfinite(a)
    if bad.any():
        k, j = (int(x) for x in np.argwhere(bad)[0])
        raise ValueError(f"{at(k, j)} is {_num(a[k, j])}, not a finite number; the runner refuses such an action")
    outside = (a < low - tol) | (a > high + tol)
    if outside.any():
        k, j = (int(x) for x in np.argwhere(outside)[0])
        raise ValueError(f"{at(k, j)} is {_num(a[k, j])}, outside the runner's limits "
                         f"[{_num(low[j])}, {_num(high[j])}]: its clamp would change it, so the replay would not send "
                         "what was recorded")
    on_limit = 0
    worst = (0.0, 0, 0)
    if len(a) > 1:
        prev, cur = a[:-1], a[1:]
        lo, hi = prev - delta, prev + delta  # DeltaLimitGuard: np.clip(data, ref - delta, ref + delta)
        over = (cur < lo - tol) | (cur > hi + tol)
        if over.any():
            i, j = (int(x) for x in np.argwhere(over)[0])
            k = i + 1
            fixed = float(np.clip(a[k, j], lo[i, j], hi[i, j]))
            raise ValueError(
                f"{at(k, j)} is {_num(a[k, j])}, {_num(a[k, j] - prev[i, j])} from the row before "
                f"({_num(prev[i, j])}); the runner's per-step limit is {_num(delta[j])} (allowed {_num(lo[i, j])} to "
                f"{_num(hi[i, j])}), so it would send {_num(fixed)} instead and the replay would not be exact")
        on_limit = int(np.count_nonzero((cur == lo) | (cur == hi)))
        frac = np.divide(np.abs(cur - prev), delta, out=np.zeros_like(cur), where=delta > 0)
        i, j = (int(x) for x in np.unravel_index(int(np.argmax(frac)), frac.shape))
        worst = (float(frac[i, j]), i + 1, j)
    ready = np.asarray(READY, dtype=np.float64)
    return {
        "rows": int(len(a)),
        "control_hz": hz,
        "duration_s": len(a) / hz,
        "on_limit": on_limit,
        "largest_step_fraction": worst[0],
        "largest_step_at": (worst[1], COLUMNS[worst[2]]),
        "row0_from_ready": {"left": float(np.max(np.abs(a[0, 0:6] - ready))),
                            "right": float(np.max(np.abs(a[0, 7:13] - ready)))},
        "grippers": (float(min(a[:, 6].min(), a[:, 13].min())) + 0.0, float(max(a[:, 6].max(), a[:, 13].max())) + 0.0),
    }


# -- the demo object a task file returns ------------------------------------------------------------------------------


class Replay:
    """A recording as the demo for both arms. yam_scripted builds an ArmPlan from it, which reads only
    ``duration``; the patched policy then plays ``rows``. It has no ``pos``: if it ever reaches the IK planner
    (the patch is missing) it says why instead of failing obscurely."""

    _kit_replay_demo = True
    label = "recorded"
    yaw = 0.0

    def __init__(self, rows: Any, control_hz: float, *, sha256: str | None = None, path: str | None = None,
                 meta: dict | None = None) -> None:
        a = np.array(rows, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != ACTION_DIM or len(a) == 0:
            raise ValueError(f"a recording is rows of {ACTION_DIM} numbers, got shape {a.shape}")
        a.setflags(write=False)
        self.rows = a
        self.control_hz = float(control_hz)
        self.steps = int(len(a))
        self.duration = self.steps / self.control_hz  # what ArmPlan.duration and yam_scripted's _end read
        self.sha256 = sha256
        self.path = path
        self.meta = dict(meta or {})

    def _no_waypoint(self) -> Any:
        raise RuntimeError(
            "kit_replay: this demo is a recorded action sequence, which yam_scripted plays only with kit_replay's "
            "patch; it is missing (the task file must import kit_replay before demo() runs)")

    pos = property(_no_waypoint)
    finger = property(_no_waypoint)

    def __repr__(self) -> str:
        name = Path(self.path).name if self.path else "rows"
        return f"Replay({name}, {self.steps} rows at {self.control_hz:g} Hz)"


def _is_replay(obj: Any) -> bool:
    return getattr(type(obj), MARKER, False) is True and hasattr(obj, "rows") and hasattr(obj, "control_hz")


def replay_demo(path: str | os.PathLike, expect_sha256: str | None = None, *,
                control_hz: float | None = None) -> dict[str, list[Replay]]:
    """A task file's demo(): ``{"left": [r], "right": [r]}`` with one Replay of the file for both arms.

    A relative path is taken from the folder of the file that calls this (the task file); a file whose sha256
    isn't ``expect_sha256`` is refused, and so is a row the runner's guards would change. The sha256 is
    :func:`demo_sha256` (line ends read as LF); the raw bytes' sha256, the rule before 18 Nov 2026, also matches."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        caller = sys._getframe(1).f_globals.get("__file__")
        p = (Path(caller).resolve().parent if caller else Path.cwd()) / p
    rows, meta = load_actions(p, control_hz=control_hz)
    if expect_sha256 is not None and str(expect_sha256).strip().lower() not in (meta["sha256"], meta["raw_sha256"]):
        raise ValueError(f"{p.name} has sha256 {meta['sha256']}, but the task file expects {expect_sha256}: the "
                         "file changed since the task file was written; set DEMO_SHA256 to the new value "
                         f"(python kit_replay.py check {p.name} prints it) only if the new file is the demo you mean")
    check_actions(rows, meta["control_hz"])
    replay = Replay(rows, meta["control_hz"], sha256=meta["sha256"], path=str(p), meta=meta)
    return {"left": [replay], "right": [replay]}


# -- the step rule, shared by the yam_scripted patch and yam_replay ---------------------------------------------------


def step_index(observation: Any, dt: float) -> int:
    """The control step an observation is for: the runner's extra["env_step"], else round(state_time / dt)."""
    extra = getattr(observation, "extra", None) or {}
    k = extra.get("env_step") if hasattr(extra, "get") else None
    if isinstance(k, (int, np.integer)) and not isinstance(k, bool):
        k = int(k)
    else:
        k = int(round(float(observation.state_time) / float(dt)))
    if k < 0:
        raise ValueError(f"kit_replay: control step {k} is negative")
    return k


def hold_steps(hold_s: float, control_hz: float) -> int:
    hold = float(hold_s)
    if not (math.isfinite(hold) and hold >= 0):
        raise ValueError(f"hold_s must be a number >= 0, got {hold_s!r}")
    return int(round(hold * float(control_hz)))


def stop_step(steps: int, hold: int) -> int:
    """The step whose action asks the runner to stop: the last row, then ``hold`` steps holding it."""
    return int(steps) - 1 + int(hold)


def make_chunk(replay: Replay, k: int, hold: int, action_type: Any, chunk_type: Any, control_hz: float) -> Any:
    """Step k's chunk, built as yam_scripted builds its own: row k (the last row once the file has run out),
    with meta {"request_stop": True} from the stop step on."""
    row = np.array(replay.rows[min(int(k), replay.steps - 1)], dtype=np.float64)
    meta = {"request_stop": True} if k >= stop_step(replay.steps, hold) else {}
    return chunk_type(actions=[action_type(data=row, meta=meta)], control_hz=float(control_hz))


def _safe(name: str) -> str:
    """inspect-robots' file-name rule for scene ids (frames._safe)."""
    safe = _SAFE_RE.sub("-", name)
    if safe != name:
        safe = f"{safe}-{zlib.crc32(name.encode()) & 0xFFFFFFFF:08x}"
    return safe


def _joint_state(observation: Any) -> np.ndarray | None:
    state = getattr(observation, "state", None) or {}
    value = state.get("joint_pos") if hasattr(state, "get") else None
    if value is None:
        return None
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    return v if v.size == ACTION_DIM else None


def write_joint_log(record: Any, log_dir: Any, run_id: Any, replay: Replay, hold: int) -> str | None:
    """After a replayed trial: write the measured joints (the observations' joint_pos) next to the run, as
    <log_dir>/kit_replay/<run>/<scene>-e<epoch>.joint_pos.csv (t = -1 is the state before step 0, t = k the state
    after step k), and put the replay's identity in record.metadata["kit_replay"] (it lands in the eval log).
    Never raises: an exception here would turn the runner's whole run into status "error"."""
    try:
        meta: dict[str, Any] = {
            "file": Path(replay.path).name if replay.path else None,
            "sha256": replay.sha256,
            "rows": replay.steps,
            "control_hz": replay.control_hz,
            "hold_steps": int(hold),
            "stop_step": stop_step(replay.steps, hold),
            "joint_log": None,
            "kit_replay_sha256": file_sha256(__file__),
        }
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, dict):
            metadata["kit_replay"] = meta
        steps = list(getattr(record, "steps", None) or [])
        if not steps:
            return None
        states = [(-1, _joint_state(steps[0].observation))]
        states += [(int(step.t), _joint_state(step.result.observation)) for step in steps]
        if all(v is None for _, v in states):
            return None
        name = f"{_safe(str(record.scene_id))}-e{int(record.epoch)}.joint_pos.csv"
        relative = Path("kit_replay") / str(run_id) / name
        path = Path(str(log_dir)) / relative
        buf = io.StringIO(newline="")
        buf.write(f"# {JOINT_LOG_FORMAT} control_hz={replay.control_hz:g} steps={len(steps)} "
                  f"record_id={getattr(record, 'record_id', None)} replay_sha256={replay.sha256}\n")
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(("t", *COLUMNS))
        for t, v in states:
            writer.writerow([t, *([""] * ACTION_DIM if v is None else [f"{x:.17g}" for x in v])])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(buf.getvalue(), encoding="utf-8")
        os.replace(tmp, path)
        meta["joint_log"] = relative.as_posix()
        return str(path)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        print(f"kit_replay: no joint log for this trial ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        return None


# -- the yam_scripted patch -------------------------------------------------------------------------------------------


def _the_replay(policy: Any) -> Replay | None:
    """The Replay a reset policy's plans carry, or None for an ordinary demo; refuses a half-recorded demo."""
    plans = getattr(policy, "_plans", None) or {}
    carried = {arm: list(getattr(plan, "waypoints", ()) or ()) for arm, plan in plans.items()}
    if not any(_is_replay(w) for ws in carried.values() for w in ws):
        return None
    left, right = carried.get("left", []), carried.get("right", [])
    if (len(left) != 1 or len(right) != 1 or left[0] is not right[0] or not _is_replay(left[0])
            or set(carried) != {"left", "right"}):
        shape = {arm: [type(w).__name__ for w in ws] for arm, ws in carried.items()}
        raise ValueError(f"a recorded demo drives both arms with one recording: demo() must return "
                         f"{{'left': [r], 'right': [r]}} as kit_replay.replay_demo() does, got {shape}")
    return left[0]


def _patch_scripted(module: Any) -> None:
    """Wrap yam_scripted's bind, reset, act and on_trial_end so a recorded demo plays its rows (idempotent)."""
    cls = getattr(module, "YamScriptedPolicy", None)
    if cls is None or PATCH_ATTR in cls.__dict__:
        return
    prev_bind, prev_reset, prev_act, prev_end = cls.bind, cls.reset, cls.act, cls.on_trial_end
    action_type, chunk_type = module.Action, module.ActionChunk  # the classes yam_scripted builds its chunks from

    def bind(self: Any, embodiment_info: Any) -> None:
        prev_bind(self, embodiment_info)
        self._kit_replay_space = getattr(embodiment_info, "action_space", None)  # the box the runner's guards use

    def reset(self: Any, scene: Any) -> None:
        self.__dict__.pop(STATE_ATTR, None)
        prev_reset(self, scene)
        replay = _the_replay(self)
        if replay is None:
            return
        if os.environ.get("KIT_RESTORE"):
            raise ValueError("kit_replay: KIT_RESTORE is set, so the scene would start from a saved state, not the "
                             "task's start the recording began from; unset KIT_RESTORE (and KIT_SAVE_STEPS)")
        hz = 1.0 / float(self._dt)
        if not math.isclose(hz, replay.control_hz, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"kit_replay: the recording is at {replay.control_hz:g} Hz but this run is at {hz:g} Hz; "
                             f"run with -E control_hz={replay.control_hz:g}")
        space = self.__dict__.get("_kit_replay_space")
        low, high, delta = runner_limits(self.info.action_space if space is None else space)
        check_actions(replay.rows, replay.control_hz, low=low, high=high, max_delta=delta)
        hold = hold_steps(self.hold_s, hz)
        setattr(self, STATE_ATTR, {"replay": replay, "hold": hold, "hz": hz})

    def act(self: Any, observation: Any) -> Any:
        state = self.__dict__.get(STATE_ATTR)
        if state is None:
            return prev_act(self, observation)
        k = step_index(observation, self._dt)
        return make_chunk(state["replay"], k, state["hold"], action_type, chunk_type, state["hz"])

    def on_trial_end(self: Any, record: Any, log_dir: Any, run_id: Any) -> None:
        prev_end(self, record, log_dir, run_id)
        state = self.__dict__.get(STATE_ATTR)
        if state is not None:
            write_joint_log(record, log_dir, run_id, state["replay"], state["hold"])

    cls.bind, cls.reset, cls.act, cls.on_trial_end = bind, reset, act, on_trial_end
    setattr(cls, PATCH_ATTR, {"bind": prev_bind, "reset": prev_reset, "act": prev_act, "on_trial_end": prev_end})


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena.scripted as it is imported (the same mechanism as labkit's)."""

    _kit_replay_finder = True

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "yam_arena.scripted":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module: Any) -> None:
            exec_module(module)
            _patch_scripted(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Patch yam_arena.scripted now if it is imported, and add the import hook if none is there. Safe to repeat."""
    module = sys.modules.get("yam_arena.scripted")
    if module is not None:
        _patch_scripted(module)
    if not any(getattr(f, "_kit_replay_finder", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


# -- reading a run's logs (stdlib + numpy) ----------------------------------------------------------------------------


def _runs(log_dir: Path) -> list[str]:
    root = log_dir / "actions"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _pick_run(log_dir: Path, run: str | None) -> tuple[str, list[str]]:
    runs = _runs(log_dir)
    if not runs:
        raise PackError(f"no runs in {log_dir} (no actions/<run> folders): pass the --log-dir of an inspect-robots run")
    if run is None:
        return runs[-1], runs
    if run not in runs:
        raise PackError(f"no run {run!r} in {log_dir}; runs there: {', '.join(runs)}")
    return run, runs


def _trials(log_dir: Path, run: str) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    for p in sorted((log_dir / "actions" / run).glob("*.jsonl")):
        m = _TRIAL_RE.match(p.stem)
        if m:
            out[(m.group("scene"), int(m.group("epoch")))] = p
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _log_is_run(data: dict, run: str) -> bool:
    stats = data.get("stats") or {}
    for key in ("records_dir", "frames_dir"):
        value = stats.get(key)
        if isinstance(value, str) and re.split(r"[\\/]", value.rstrip("/\\"))[-1] == run:
            return True
    for sample in data.get("samples") or []:
        for meta in sample.get("trial_metadata") or []:
            if isinstance(meta, dict) and str(meta.get("actions", "")).startswith(f"actions/{run}/"):
                return True
        for rd in sample.get("record_dirs") or []:
            if isinstance(rd, str) and rd.startswith(f"records/{run}/"):
                return True
    return False


def _epoch_at(sample: dict, i: int) -> int:
    for key, strip in (("record_dirs", ""), ("trial_metadata", ".jsonl")):
        values = sample.get(key) or []
        value = values[i] if i < len(values) else None
        if key == "trial_metadata":
            value = value.get("actions") if isinstance(value, dict) else None
        if isinstance(value, str):
            stem = value.rsplit("/", 1)[-1]
            stem = stem[: -len(strip)] if strip and stem.endswith(strip) else stem
            m = _TRIAL_RE.match(stem)
            if m:
                return int(m.group("epoch"))
    return i


def _logs_of_run(log_dir: Path, run: str, *, live: bool = False) -> list[tuple[Path, dict]]:
    """The run's eval logs, or with ``live`` its live logs (<task>_<id>.live.json: the runner's snapshot while a run
    is going, with the same eval block but no scores yet, deleted when the eval log is written and left behind when
    the run is killed)."""
    found: list[tuple[Path, dict]] = []
    for path in sorted(log_dir.glob("*.json")):
        if path.name.endswith(".live.json") != live:
            continue
        data = _read_json(path)
        if (isinstance(data, dict) and isinstance(data.get("samples"), list) and isinstance(data.get("eval"), dict)
                and _log_is_run(data, run)):
            found.append((path, data))
    return found


def _run_identity(log_dir: Path, run: str) -> dict | None:
    """What the run's eval log (else its live log) says the run was: environment_id (yam_arena's ends in
    -task-<task file name>), environment_revision (the first 12 hex digits of the task file's sha256), max_steps
    and the control rate, plus which log said so. None when neither log is there."""
    for live in (False, True):
        found = _logs_of_run(log_dir, run, live=live)
        if found:
            path, data = found[-1]
            ev = data["eval"]
            return {"log": path.name, "environment_id": ev.get("environment_id"),
                    "environment_revision": ev.get("environment_revision"), "max_steps": ev.get("max_steps"),
                    "control_hz": (ev.get("embodiment_info") or {}).get("control_hz")}
    return None


def _task_of(environment_id: Any) -> str | None:
    """The task file (its name without .py) a yam_arena environment_id names, or None when it names none."""
    _, sep, name = str(environment_id or "").partition("-task-")
    return name if sep and name else None


def _trial_rate(record: dict | None, identity: dict | None) -> float | None:
    """A trial's control rate: its record.json's, else the run's eval (or live) log's embodiment_info."""
    hz = (((record or {}).get("configuration") or {}).get("embodiment") or {}).get("control_hz")
    if hz is None and identity:
        hz = identity.get("control_hz")
    return hz


def _eval_index(log_dir: Path, run: str) -> dict[tuple[str, int], dict]:
    """What the run's eval log says about each trial, keyed by (file-safe scene id, epoch)."""
    out: dict[tuple[str, int], dict] = {}
    for path, data in _logs_of_run(log_dir, run):
        ev = data["eval"]
        for sample in data["samples"]:
            if not isinstance(sample, dict):
                continue
            epochs = sample.get("epochs") or []
            reasons = sample.get("termination_reasons") or []
            metas = sample.get("trial_metadata") or []
            for i, scores in enumerate(epochs):
                key = (_safe(str(sample.get("scene_id"))), _epoch_at(sample, i))
                scores = scores if isinstance(scores, dict) else {}
                out[key] = {
                    "log": path.name,
                    "scores": scores,
                    "score": scores.get("task_score"),
                    "termination": reasons[i] if i < len(reasons) else None,
                    "trial_metadata": metas[i] if i < len(metas) and isinstance(metas[i], dict) else {},
                    "instruction": sample.get("instruction"),
                    "policy": ev.get("policy"),
                    "frames_dir": (data.get("stats") or {}).get("frames_dir"),
                    "status": data.get("status"),
                }
    return out


def _record_json(log_dir: Path, run: str, trial_id: str) -> dict | None:
    data = _read_json(log_dir / "records" / run / trial_id / "record.json")
    return data if isinstance(data, dict) else None


def _record_score(record: dict | None) -> tuple[float | None, str]:
    """task_score from a record alone: a trial that ended with success scores 1.0 by task_score's own rule (the
    record keeps the termination, not the score); anything else needs the eval log."""
    if record is None:
        return None, "no eval log and no record.json for it"
    term = record.get("termination") or {}
    rubrics = (record.get("configuration") or {}).get("rubrics") or []
    scored_by_task_score = any(isinstance(r, dict) and r.get("scorer") == "task_score" for r in rubrics)
    if (record.get("status") == "success" and term.get("terminated") is True and term.get("reason") == "success"
            and scored_by_task_score):
        return 1.0, "record.json: the trial ended with success, which task_score scores 1.0 (no eval log yet)"
    return None, (f"no eval log for this run yet (the runner writes it when the run ends), and record.json alone "
                  f"can't give the score of a trial that ended with {term.get('reason')!r}")


def _guard_config(record: dict | None) -> tuple[list[str], dict]:
    guard = ((record or {}).get("configuration") or {}).get("guard") or {}
    names = list(guard.get("names") or [])
    configs: dict[str, Any] = {}
    for name, member in zip(names, guard.get("guards") or []):
        configs[name] = (member or {}).get("config") or {}
    return names, configs


def _guard_is_default(record: dict | None) -> tuple[bool, str]:
    names, configs = _guard_config(record)
    if tuple(names) != GUARD_NAMES:
        return False, f"guards {names or 'none'}, not the default {list(GUARD_NAMES)}"
    clamp, delta = configs.get("clamp") or {}, configs.get("delta-limit") or {}
    if clamp.get("low") != RUNNER_LOW.tolist() or clamp.get("high") != RUNNER_HIGH.tolist():
        return False, "the clamp box differs from yam_arena's joint limits"
    if delta.get("max_delta") != RUNNER_MAX_DELTA.tolist():
        return False, f"the per-step limit differs from the default (max_delta {delta.get('max_delta')})"
    return True, "clamp to the joint limits, then the default per-step limit (5% of each range)"


def _frames_dir(log_dir: Path, run: str, entry: dict | None) -> Path:
    stored = (entry or {}).get("frames_dir")
    if isinstance(stored, str) and stored:
        for candidate in (Path(stored), log_dir / "frames" / re.split(r"[\\/]", stored.rstrip("/\\"))[-1]):
            if candidate.is_dir():
                return candidate
    return log_dir / "frames" / run


def _frame_component(name: str) -> str:
    return "".join(chr(b) if b in _FRAME_OK else f"%{b:02X}" for b in name.encode("utf-8"))


def _camera_source(frames_dir: Path, trial_id: str, camera: str = TOP_CAMERA) -> tuple[str, Any] | None:
    """('mp4', path) for the camera's video as `inspect-robots video` writes it, else ('frames', [paths]) for its
    stored frames, else None. Stream names: ~f1~<trial>~<camera> (inspect-robots 1.0), or <trial>_<camera>."""
    stems = (f"~f1~{_frame_component(trial_id)}~{_frame_component(camera)}", f"{trial_id}_{camera}")
    for stem in stems:
        if (frames_dir / f"{stem}.mp4").is_file():
            return "mp4", frames_dir / f"{stem}.mp4"
    for stem in stems:
        pattern = re.compile(re.escape(stem) + r"_(\d{6,})\.npy$")
        frames = []
        if frames_dir.is_dir():
            for p in frames_dir.iterdir():
                m = pattern.match(p.name)
                if m:
                    frames.append((int(m.group(1)), p))
        if frames:
            return "frames", [p for _, p in sorted(frames)]
    return None


# -- media ------------------------------------------------------------------------------------------------------------


def _mp4_boxes(path: Path) -> list[str]:
    """The MP4's top-level box types in file order (enough to see where moov sits)."""
    out: list[str] = []
    size_total = path.stat().st_size
    with path.open("rb") as f:
        pos = 0
        while pos + 8 <= size_total:
            f.seek(pos)
            size, kind = struct.unpack(">I4s", f.read(8))
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            elif size == 0:
                size = size_total - pos
            if size < 8:
                break
            out.append(kind.decode("latin-1"))
            pos += size
    return out


def faststart(path: Path) -> bool:
    boxes = _mp4_boxes(path)
    return "moov" in boxes and "mdat" in boxes and boxes.index("moov") < boxes.index("mdat")


def _probe(ffprobe: str, path: Path) -> dict:
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt,width,height", "-of", "json", str(path)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise PackError(f"ffprobe can't read {path.name}: {r.stderr.strip()[-400:]}")
    streams = (json.loads(r.stdout or "{}").get("streams")) or []
    if not streams:
        raise PackError(f"{path.name} has no video stream")
    return streams[0]


def _ffmpeg(ffmpeg: str, args: list[str], stdin: Any = None) -> None:
    """Run ffmpeg (stderr to a file, never a pipe, so it can't fill up and stall); ``stdin`` is an iterable of
    byte chunks to feed it."""
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y", *args]
    with tempfile.TemporaryFile() as err:
        if stdin is None:
            code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err,
                                  timeout=1800).returncode
        else:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
            try:
                for chunk in stdin:
                    proc.stdin.write(chunk)
            except BrokenPipeError:
                pass  # ffmpeg stopped reading; its exit code and message say why
            except BaseException:
                proc.kill()
                raise
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                code = proc.wait()
        err.seek(0)
        tail = err.read().decode("utf-8", "replace").strip()[-600:]
    if code != 0:
        raise PackError(f"ffmpeg failed (exit {code}): {tail or 'no message'}")


_EVEN = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
NO_FFMPEG = ("ffmpeg and ffprobe are needed for demo.mp4 and setup.png and weren't found on PATH (on a Mac: brew "
             "install ffmpeg; on Ubuntu: sudo apt install ffmpeg)")


def _frames_to_mp4(ffmpeg: str, frames: list[Path], fps: float, out: Path) -> None:
    """Encode stored .npy frames (uint8 RGB) to H.264 yuv420p with the index at the front."""
    first = None
    for p in frames:
        a = np.load(p)
        if a.size:
            first = a
            break
    if first is None or first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] not in (3, 4):
        raise PackError(f"the stored frames ({frames[0].name}, ...) aren't uint8 RGB images")
    height, width = first.shape[:2]

    def chunks():
        for p in frames:
            a = np.load(p)
            if a.size == 0:
                continue  # inspect-robots skips empty warm-up frames too
            if a.shape[:2] != (height, width):
                raise PackError(f"frame size changed at {p.name}")
            yield np.ascontiguousarray(a[..., :3], dtype=np.uint8).tobytes()

    _ffmpeg(ffmpeg, ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
                     "-vf", _EVEN, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            stdin=chunks())


def _png_size(path: Path) -> tuple[int, int] | None:
    head = path.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def make_media(source: tuple[str, Any], workdir: Path, *, fps: float, max_video_bytes: int = MAX_VIDEO_BYTES,
               ffmpeg: str | None = None, ffprobe: str | None = None) -> dict:
    """demo.mp4 (H.264, yuv420p, index at the front, no metadata) and setup.png (its first frame) in ``workdir``.

    An H.264 yuv420p video is copied stream for stream (no re-encoding); anything else, or stored frames, is
    encoded with libx264. Refuses a video over the program's size limit."""
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise PackError(NO_FFMPEG)
    demo, png = workdir / "demo.mp4", workdir / "setup.png"
    kind, src = source
    if kind == "frames":
        _frames_to_mp4(ffmpeg, src, fps, demo)
        how = f"encoded from {len(src)} stored frames (H.264, yuv420p)"
    else:
        info = _probe(ffprobe, src)
        if info.get("codec_name") == "h264" and info.get("pix_fmt") == "yuv420p":
            _ffmpeg(ffmpeg, ["-i", str(src), "-map", "0:v:0", "-c", "copy", "-map_metadata", "-1",
                             "-movflags", "+faststart", str(demo)])
            how = "copied without re-encoding (already H.264 yuv420p); index moved to the front, metadata dropped"
        else:
            _ffmpeg(ffmpeg, ["-i", str(src), "-map", "0:v:0", "-vf", _EVEN, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-map_metadata", "-1", "-movflags", "+faststart", str(demo)])
            how = f"re-encoded to H.264 yuv420p (it was {info.get('codec_name')} {info.get('pix_fmt')})"
    size = demo.stat().st_size
    if size > max_video_bytes:
        raise PackError(f"demo.mp4 would be {size / 1e6:.1f} MB, over the program's {max_video_bytes / 1e6:g} MB "
                        "limit for a video (tools/upload_media.py): it can't be uploaded as it is")
    final = _probe(ffprobe, demo)
    if final.get("codec_name") != "h264" or final.get("pix_fmt") != "yuv420p" or not faststart(demo):
        raise PackError(f"demo.mp4 came out as {final.get('codec_name')} {final.get('pix_fmt')} "
                        f"(index at the front: {faststart(demo)}), not H.264 yuv420p with +faststart")
    _ffmpeg(ffmpeg, ["-i", str(demo), "-frames:v", "1", "-map_metadata", "-1", str(png)])
    if _png_size(png) is None:
        raise PackError("setup.png didn't come out as a PNG")
    if png.stat().st_size > MAX_SCREENSHOT_BYTES:
        raise PackError(f"setup.png is {png.stat().st_size / 1e6:.1f} MB, over the program's 10 MB limit")
    return {"demo": demo, "setup": png, "how": how, "bytes": size, "width": final.get("width"),
            "height": final.get("height")}


# -- the task folder --------------------------------------------------------------------------------------------------


def find_task_file(folder: Path) -> Path:
    """<folder name with - as _>.py, else the one .py that isn't a test, labkit.py or kit_replay.py."""
    main = folder / f"{folder.name.replace('-', '_')}.py"
    if main.is_file():
        return main
    found = [p for p in sorted(folder.glob("*.py"))
             if not p.name.startswith("test_") and p.name not in ("labkit.py", "kit_replay.py")]
    if len(found) != 1:
        raise PackError(f"can't tell which .py in {folder} is the task file: {[p.name for p in found]}")
    return found[0]


def run_limit_steps(folder: Path) -> tuple[int | None, str]:
    """The task.json run limit in steps ('2,400 steps (2 minutes at 20 Hz)' -> 2400), as check_task.py reads it."""
    data = _read_json(folder / "task.json")
    text = str((data or {}).get("run_limit", "")) if isinstance(data, dict) else ""
    m = re.search(r"([\d,]+)\s*steps", text)
    if m:
        return int(m.group(1).replace(",", "")), "task.json run_limit"
    return None, "task.json has no run limit in steps"


def blind_words(path: Path | None = None) -> tuple[list[str], str]:
    """The kit's blind-review word list (one per line, '#' for comments), found next to the kit."""
    if path is None:
        here = Path(__file__).resolve()
        candidates = [d / "blind_words.txt" for d in list(here.parents)[:4]]
        path = next((c for c in candidates if c.is_file()), None)
    if path is None or not Path(path).is_file():
        return [], "no blind_words.txt found (pass --blind-words): only paths were checked"
    words = [w.strip() for w in Path(path).read_text(encoding="utf-8").splitlines()
             if w.strip() and not w.strip().startswith("#")]
    return words, f"checked against {len(words)} words in blind_words.txt"


def blind_hits(text: str, words: list[str]) -> list[str]:
    hits = [repr(w) for w in words if re.search(re.escape(w), text, re.I)]
    if _PATH_RE.search(text):
        hits.append("an absolute path")
    return hits


# -- the scene a trial was recorded on (keyboard_run.py stamps every trial) -------------------------------------------

SCENE_TEXT = (".py", ".json", ".usda", ".obj", ".mtl", ".txt")  # hashed with CRLF read as LF (the asset pins' rule)


def _test_file(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def scene_files(folder: str | os.PathLike, task_file: str | os.PathLike | None = None) -> dict[str, str]:
    """The files a task folder's scene is built from, relative POSIX path -> sha256: labkit.py and every other .py
    except the task file itself (its sha256 is the run's environment_revision), tests and kit_replay.py (which pack
    writes after the recording), plus every file under assets/ (the frozen assets). Dotfiles and __pycache__ are left
    out. Text files (SCENE_TEXT) are hashed with CRLF read as LF, so a checkout that changes line ends still matches.
    keyboard_run.py stamps every trial with the same rule (trial_metadata["kit_scene"]["files"]); pack compares."""
    folder = Path(folder)
    top = {Path(task_file).name if task_file is not None else None, "kit_replay.py"}
    out: dict[str, str] = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts) or not p.is_file():
            continue
        if rel.parts[0] != "assets" and (p.suffix != ".py" or _test_file(p.name)
                                         or (len(rel.parts) == 1 and p.name in top)):
            continue
        data = p.read_bytes()
        out[rel.as_posix()] = lf_sha256(data) if p.suffix.lower() in SCENE_TEXT else hashlib.sha256(data).hexdigest()
    return out


def _names(paths: list[str], most: int = 6) -> str:
    shown = ", ".join(paths[:most])
    return shown + (f" and {len(paths) - most} more" if len(paths) > most else "")


def _files_summary(paths) -> str:
    """'labkit.py, assets/apple (20 files)': the stamped files, the .py files first, an asset folder's counted as one."""
    groups: dict[str, int] = {}
    for rel in sorted(paths, key=lambda r: (r.startswith("assets/"), r)):
        parts = rel.split("/")
        key = "/".join(parts[:2]) if parts[0] == "assets" and len(parts) > 2 else rel
        groups[key] = groups.get(key, 0) + 1
    return ", ".join(f"{k} ({n} files)" if n > 1 else k for k, n in groups.items()) or "no files"


def compare_stamp(stamp: Any, folder: Path, task_file: Path) -> dict:
    """A trial's scene stamp (trial_metadata["kit_scene"], None for a trial without one) against the task folder
    now: {"stamped": bool, "why": why there is nothing to compare (or None), "library": the library assets it placed,
    "changed", "added", "removed": scene files (scene_files) that differ, "files": the folder's scene files}."""
    now = scene_files(folder, task_file)
    out: dict[str, Any] = {"stamped": False, "why": None, "library": [], "changed": [], "added": [], "removed": [],
                           "files": now}
    if stamp is None:
        out["why"] = "it carries no scene stamp (recorded by an older keyboard_run, or no eval log of its run is here)"
        return out
    files, library = (stamp.get("files"), stamp.get("library_assets")) if isinstance(stamp, dict) else (None, None)
    if isinstance(stamp, dict) and stamp.get("error"):
        out["why"] = f"keyboard_run couldn't stamp its scene ({stamp['error']})"
        return out
    if (not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
            or not isinstance(library, list)):
        out["why"] = f"its scene stamp can't be read ({str(stamp)[:120]})"
        return out
    out.update(stamped=True, library=sorted(str(a) for a in library),
               changed=sorted(k for k in files.keys() & now.keys() if files[k] != now[k]),
               added=sorted(now.keys() - files.keys()), removed=sorted(files.keys() - now.keys()))
    return out


def scene_problems(scene: dict, trial: int, task_dir: Path) -> list[str]:
    """Why a trial's stamp (compare_stamp) says it wasn't recorded on this task folder's scene; empty when it was."""
    problems = []
    if scene["library"]:
        problems.append(f"trial {trial} was recorded on a library scene (library assets {', '.join(scene['library'])}, "
                        "spawned by keyboard_run's hook with -P library_scene=1): official runs can't import "
                        "assetlib, so they never build that scene; freeze it into the task folder (kit/tools/"
                        "freeze.py <scene> --to <task folder>) and record on the frozen folder")
    differ = [f"{what} {_names(scene[what])}" for what in ("changed", "added", "removed") if scene[what]]
    if differ:
        problems.append(f"trial {trial} was recorded on scene files that differ from {task_dir.name}'s now ("
                        + "; ".join(differ) + "; keyboard_run stamped them when its run started): the replay builds "
                        "the folder as it is now, so it may not repeat the trial")
    return problems


_TO_PATH = ("Path", "pathlib.Path")
_TO_STR = ("str", "os.fspath", "os.path.abspath", "os.path.realpath", "os.path.normpath")


def _dotted(node: ast.AST) -> str | None:
    """'os.path.dirname' for the expression os.path.dirname, None for anything but a dotted name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _levels_up(node: ast.AST, names: dict[str, tuple[int, bool]]) -> tuple[int, bool] | None:
    """(how many folders up from the task file, whether it is a str) for a path expression built from __file__
    with the usual calls (0 is the file itself, 1 its folder), following module-level names already worked out;
    None for anything else. Only a str in sys.path is searched: the import system skips a Path there."""
    if isinstance(node, ast.Name):
        return (0, True) if node.id == "__file__" else names.get(node.id)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        got = _levels_up(node.value, names)
        return None if got is None else (got[0] + 1, False)
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int and node.slice.value >= 0):
        got = _levels_up(node.value.value, names)
        return None if got is None else (got[0] + 1 + node.slice.value, False)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("resolve", "absolute") and not node.args:
            got = _levels_up(node.func.value, names)
            return None if got is None else (got[0], False)
        if len(node.args) == 1 and not node.keywords:
            name = _dotted(node.func)
            got = _levels_up(node.args[0], names) if name in (*_TO_PATH, *_TO_STR, "os.path.dirname") else None
            if got is not None:
                return (got[0] + (name == "os.path.dirname"), name not in _TO_PATH)
    return None


def _module_level(tree: ast.Module) -> Any:
    """(top-level statement, statement) for each statement that runs when the module loads, in order: the module
    body and the bodies of its top-level if, try and with blocks (not function or class bodies)."""
    def inner(stmts: list[ast.stmt]) -> Any:
        for s in stmts:
            yield s
            if isinstance(s, ast.If):
                yield from inner(s.body)
                yield from inner(s.orelse)
            elif isinstance(s, ast.With):
                yield from inner(s.body)
            elif isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
                yield from inner(s.body)
                for handler in s.handlers:
                    yield from inner(handler.body)
                yield from inner(s.orelse)
                yield from inner(s.finalbody)

    for top in tree.body:
        for s in inner([top]):
            yield top, s


def assetlib_imports(tree: ast.AST) -> list[int]:
    """The lines where a module's syntax tree imports the kit's assetlib (import assetlib, from assetlib import ...,
    import assetlib.x, importlib.import_module("assetlib"), __import__("assetlib")), anywhere in the file."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
        elif (isinstance(node, ast.Call) and _dotted(node.func) in ("importlib.import_module", "import_module",
                                                                       "__import__")
              and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names = [node.args[0].value]
        if any(n == "assetlib" or n.startswith("assetlib.") for n in names):
            lines.add(node.lineno)
    return sorted(lines)


def _task_file_state(task_file: Path) -> dict:
    """What the task file already has of the replay lines, read from its syntax tree: only statements that run when
    the module loads count, and a sys.path insert counts only when it adds the file's own folder. ``assetlib``: the
    lines that import the kit's assetlib (a library scene; a file Python can't parse counts if it names assetlib)."""
    source = task_file.read_text(encoding="utf-8")
    state: dict[str, Any] = {"insert_line": None, "sys_line": None, "path_line": None, "kit_line": None,
                             "head_end": 0, "head_kind": "top", "sha": None, "sha_line": None, "demo_line": None,
                             "demo_replays": False, "assetlib": [], "unparsed": None}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        state["unparsed"] = f"line {exc.lineno}: {exc.msg}"
        state["assetlib"] = [i for i, line in enumerate(source.splitlines(), 1) if re.search(r"\bassetlib\b", line)]
        return state
    state["assetlib"] = assetlib_imports(tree)
    for i, node in enumerate(tree.body):  # the docstring and the imports at the top: where new imports go
        if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str):
            state["head_end"], state["head_kind"] = node.end_lineno, "docstring"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            state["head_end"], state["head_kind"] = node.end_lineno, "imports"
        else:
            break
    names: dict[str, tuple[int, bool]] = {}
    for top, node in _module_level(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            up = _levels_up(node.value, names)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    if up is None:
                        names.pop(target.id, None)
                    else:
                        names[target.id] = up
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for module, key in (("sys", "sys_line"), ("kit_replay", "kit_line")):
                    if alias.name == module and alias.asname in (None, module) and state[key] is None:
                        state[key] = node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "pathlib" and not node.level:
            if any(a.name == "Path" and a.asname in (None, "Path") for a in node.names) and state["path_line"] is None:
                state["path_line"] = node.lineno
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and state["insert_line"] is None:
            call, fn = node.value, _dotted(node.value.func)
            arg = (call.args[1] if fn == "sys.path.insert" and len(call.args) == 2
                   else call.args[0] if fn == "sys.path.append" and len(call.args) == 1 else None)
            if arg is not None and _levels_up(arg, names) == (1, True):  # the file's own folder, as a str
                state["insert_line"] = top.end_lineno  # after the whole top-level block that holds it
    for node in tree.body:
        named = isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEMO_SHA256"
                                                      for t in node.targets)
        if named:
            state["sha_line"] = node.lineno
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                state["sha"] = node.value.value
        if isinstance(node, ast.FunctionDef) and node.name == "demo":
            state["demo_line"] = node.lineno
            state["demo_replays"] = "replay_demo" in (ast.get_source_segment(source, node) or "")
    return state


def task_file_lines(task_file: Path, sha: str, csv_name: str = DEMO_FILE) -> list[str]:
    """What to add to (or change in) the task file so its demo() replays the file, and where it goes: the import
    must run after the file's own folder is on sys.path, which must come after sys and Path are imported."""
    state = _task_file_state(task_file)
    name = task_file.name
    insert, kit = state["insert_line"], state["kit_line"]
    ready = insert is not None and kit is not None and kit > insert  # it can import kit_replay
    if ready and state["demo_replays"] and state["sha"] == sha:
        return [f"{name} already replays this file (DEMO_SHA256 matches): nothing to change."]
    if ready and state["demo_replays"] and state["sha"] is not None:
        return [f"In {name}, change DEMO_SHA256 (line {state['sha_line']}; it names the previous file) to:", "",
                f'DEMO_SHA256 = "{sha}"']
    if insert is not None:
        after = insert
        out = [f"Add these lines to {name} after line {insert}, where it puts its own folder on sys.path:", ""]
    else:
        after = state["head_end"]
        where = f"after line {after}, the end of its {state['head_kind']}" if after else "at its top"
        out = [f"Add these lines to {name} {where}:", ""]
        if state["sys_line"] is None or state["sys_line"] > after:
            out.append("import sys")
        if state["path_line"] is None or state["path_line"] > after:
            out.append("from pathlib import Path")
        out += ["", "sys.path.insert(0, str(Path(__file__).resolve().parent))  # kit_replay.py sits next to this file"]
    if kit is None or kit <= after:
        out.append("import kit_replay  # noqa: E402")
    out += ["", f'DEMO_SHA256 = "{sha}"', "", "", "def demo():",
            f'    """A recorded run of the task, replayed open loop from {csv_name}."""',
            f'    return kit_replay.replay_demo("{csv_name}", expect_sha256=DEMO_SHA256)']
    if insert is None:
        out += ["", "(The file doesn't put its own folder on sys.path when it loads; the import needs it, as "
                    "labkit's does.)"]
    else:
        out += ["", "(It puts its own folder on sys.path at that line, so the import finds kit_replay.py there.)"]
    if state["unparsed"]:
        out.append(f"(Python can't parse {name} as it is ({state['unparsed']}), so these lines assume nothing.)")
    if kit is not None and kit <= after:
        out.append(f"Remove the 'import kit_replay' at line {kit}: it runs before the folder is on sys.path.")
    if state["sha_line"] is not None:
        out.append(f"Remove the old DEMO_SHA256 at line {state['sha_line']}.")
    if state["demo_line"] is not None:
        out.append(f"Remove the old demo() at line {state['demo_line']}: a task file has one demo().")
    return out


def _inside_git(path: Path) -> Path | None:
    for d in (path, *path.parents):
        if (d / ".git").exists():
            return d
    return None


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.kit_replay.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def pack_trial(log_dir: str | os.PathLike, trial: int, task: str | os.PathLike, media: str | os.PathLike | None = None,
               *, any_score: bool = False, other_task: bool = False, over_limit: bool = False,
               scene_changed: bool = False, run: str | None = None, hold_s: float = DEFAULT_HOLD_S,
               blind_words_file: str | os.PathLike | None = None, max_video_bytes: int = MAX_VIDEO_BYTES,
               out: Any = print) -> dict:
    """Turn the page's trial N of a run into the submission's pieces (see the module docstring and the kit's
    Isaac README). Writes demo_actions.csv and kit_replay.py into the task folder, and nothing else there;
    demo.mp4 and setup.png into the media folder. Everything is checked before anything is written.

    Refuses a task file that imports assetlib (always: official runs can't load it), a trial without full marks
    (``any_score`` packs it anyway), one longer than the task's run limit (``over_limit``), one recorded on another
    task file (``other_task``), and one whose scene stamp shows a library scene or scene files that differ from the
    task folder's (``scene_changed``)."""
    log_dir = Path(log_dir).expanduser().resolve()
    task_dir = Path(task).expanduser().resolve()
    if not log_dir.is_dir():
        raise PackError(f"no log folder at {log_dir}")
    if not task_dir.is_dir():
        raise PackError(f"no task folder at {task_dir}")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
        raise PackError(f"--trial counts from 1, as the page numbers trials; got {trial!r}")
    epoch = trial - 1
    run, runs = _pick_run(log_dir, run)
    trials = _trials(log_dir, run)
    matches = sorted((scene, p) for (scene, e), p in trials.items() if e == epoch)
    if not matches:
        have = sorted({e + 1 for (_, e) in trials})
        raise PackError(f"run {run} has no trial {trial} (epoch {epoch}); its trials: "
                        f"{', '.join(map(str, have)) or 'none'}")
    if len(matches) > 1:
        raise PackError(f"run {run} has trial {trial} for several scenes ({[s for s, _ in matches]}); "
                        "pack works on single-scene runs (the Station's)")
    scene, actions_path = matches[0]
    trial_id = actions_path.stem
    record = _record_json(log_dir, run, trial_id)
    entry = _eval_index(log_dir, run).get((scene, epoch))
    identity = _run_identity(log_dir, run)

    # the score: the eval log's task_score, else what record.json implies
    if entry is not None and entry["score"] is not None:
        score, score_from = float(entry["score"]), f"task_score in {entry['log']}"
    elif entry is not None and entry["scores"]:
        score, score_from = None, f"{entry['log']} scored it with {sorted(entry['scores'])}, not task_score"
    elif entry is not None:
        score, score_from = None, f"{entry['log']} has no score for it (the trial didn't finish: cancelled or error)"
    else:
        score, score_from = _record_score(record)
    full = score is not None and score >= FULL_MARKS
    if not full and not any_score:
        said = "has no score" if score is None else f"scored {score:g}"
        raise PackError(f"trial {trial} {said} ({score_from}), not full marks; the submission's demo must score 1. "
                        "Pick a full-score trial, or pass --any to pack it anyway")

    # the rate: record.json's, else the eval log's (an action log carries none)
    rate = _trial_rate(record, identity)
    if rate is None:
        raise PackError(f"trial {trial}'s action log carries no control rate, and there is no record.json for it "
                        f"(records/{run}/{trial_id}/) and no eval log that gives one: copy the run's records folder "
                        "back from the GPU machine too")
    rows, meta = load_actions(actions_path, control_hz=rate)
    hz = meta["control_hz"]
    try:
        summary = check_actions(rows, hz)
    except ValueError as exc:
        raise PackError(f"trial {trial}'s actions can't replay exactly: {exc}") from None
    if not math.isclose(hz, DEFAULT_HZ):
        raise PackError(f"trial {trial} ran at {hz:g} Hz; Robocurve runs are at 20 Hz (-E control_hz=20)")

    task_file = find_task_file(task_dir)
    imports = _task_file_state(task_file)["assetlib"]
    if imports:
        at = ", ".join(map(str, imports))
        raise PackError(f"{task_file.name} imports assetlib (line {at}): official runs (the replay, Dream-1, the "
                        "reviewers) can't import it, so the task file wouldn't even load there and "
                        f"{task_dir.name} can't be the submission. Freeze it into a new task folder (kit/tools/"
                        f"freeze.py {task_dir} --to ~/robocurve/benchmark-tasks/tasks/<slug>; freeze never writes "
                        "into the scene's own folder), record on that folder, and pack its trial with --task set to "
                        "it")
    limit, limit_from = run_limit_steps(task_dir)
    notes: list[str] = []
    if limit is None:
        notes.append(f"{limit_from}; the commands below use {DEFAULT_RUN_LIMIT} steps, the template's limit")
        limit = DEFAULT_RUN_LIMIT
    if len(rows) > limit:
        if not over_limit:
            raise PackError(f"trial {trial} took {len(rows)} steps, more than the task's run limit of {limit} "
                            f"({limit_from}): the official run would cut it short (--over-limit packs it anyway)")
        notes.append(f"trial {trial} took {len(rows)} steps, more than the task's run limit of {limit} (packed with "
                     "--over-limit): the official run cuts it short unless task.json's run_limit is raised first")

    # the task file the trial was recorded on: the run's environment_id names it, environment_revision dates it
    now = file_sha256(task_file)[:12]
    env_id = (identity or {}).get("environment_id")
    revision = (identity or {}).get("environment_revision")
    if not env_id:
        notes.append(f"pack couldn't check that trial {trial} was recorded on {task_file.name}: no eval log or live "
                     f"log of run {run} is here to name the task file it ran (environment_id); be sure it was")
    else:
        recorded_on = _task_of(env_id)
        what = f"task {recorded_on!r}" if recorded_on else f"no task file ({env_id})"
        said = f"{identity['log']}: environment_id, revision {revision}"
        if recorded_on != task_file.stem and revision != now:
            if not other_task:
                raise PackError(f"trial {trial} was recorded on {what} ({said}), not {task_file.stem!r} (revision "
                                f"{now}): the wrong --task folder? If it is the same scene renamed and edited, "
                                "--other-task packs it anyway")
            notes.append(f"trial {trial} was recorded on {what}, not on {task_file.name} (packed with --other-task): "
                         "the replay matches only if the objects and their start are the same")
        elif recorded_on != task_file.stem:
            notes.append(f"the trial was recorded on {recorded_on}.py, a file identical to {task_file.name}")
        elif revision and revision != now:
            notes.append(f"{task_file.name} has changed since the trial was recorded (revision {revision} then, "
                         f"{now} now): the replay is only valid if the objects and their start are unchanged "
                         "(adding the demo lines changes the revision too, which is fine)")
    # the scene files the trial was recorded on: keyboard_run's stamp against the folder now
    scene = compare_stamp((entry or {}).get("trial_metadata", {}).get("kit_scene"), task_dir, task_file)
    problems = scene_problems(scene, trial, task_dir)
    if problems and not scene_changed:
        raise PackError("; ".join(problems) + ". Record again on this folder, or pass --scene-changed to pack it "
                                              "anyway (then verify must PASS)")
    if problems:
        notes.append("packed with --scene-changed: " + "; ".join(problems) + ". The replay is valid only if verify "
                                                                             "PASSes")
        scene_line = "its stamp doesn't match this folder (packed with --scene-changed; see the notes)"
    elif scene["stamped"]:
        scene_line = f"the trial's stamp matches this folder ({_files_summary(scene['files'])})"
    else:
        scene_line = f"not checked: {scene['why']}"
        notes.append(f"pack couldn't check trial {trial}'s scene files against {task_dir.name}'s: {scene['why']}; "
                     "verify's score decides")
    hold = hold_steps(hold_s, hz)
    if len(rows) + hold > limit:
        notes.append(f"{len(rows)} rows + {hold} hold steps is more than the run limit ({limit}): harmless when the "
                     "trial ends by success first, as a full-score recording usually does")
    ended = (entry or {}).get("termination") or ((record or {}).get("termination") or {}).get("reason")
    if ended != "success":
        notes.append(f"the trial ended with {ended!r}, not success: the replay then holds the last row "
                     f"{hold_s:g} s longer than the recording did before it stops, so its score can differ; "
                     "verify reports it")

    # the media folder: never in the task folder or any git repository
    media_dir = (Path(media).expanduser() if media is not None
                 else Path.home() / "robocurve" / "runs" / task_dir.name / "upload").resolve()
    if media_dir == task_dir or task_dir in media_dir.parents:
        raise PackError(f"--media {media_dir} is inside the task folder; demo.mp4 and setup.png are uploaded, "
                        "not committed")
    repo = _inside_git(media_dir)
    if repo is not None:
        raise PackError(f"--media {media_dir} is inside the git repository {repo}; put the media outside any repo")

    # what goes into the task folder, checked for blind review before anything is written
    data_csv = csv_bytes(rows, hz)
    sha = demo_sha256(data_csv)
    me = Path(__file__).resolve()
    data_me = me.read_bytes()
    words, words_note = blind_words(Path(blind_words_file) if blind_words_file else None)
    for name, data in ((DEMO_FILE, data_csv), ("kit_replay.py", data_me)):
        hits = blind_hits(data.decode("utf-8", "replace"), words)
        if hits:
            raise PackError(f"{name} would carry {', '.join(hits)} into the task folder (blind review)")
    again, _ = _load_csv(Path(DEMO_FILE), data_csv.decode("utf-8"))
    if not np.array_equal(again, rows):
        raise PackError("the CSV didn't read back bit for bit")  # can't happen with %.17g; checked anyway

    # the video
    frames_dir = _frames_dir(log_dir, run, entry)
    source = _camera_source(frames_dir, trial_id)
    if source is None:
        raise PackError(f"no top-camera video or frames for trial {trial} in {frames_dir}: render the videos with "
                        "`inspect-robots video <eval log>` where the run is, or copy the run's frames folder too")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise PackError(NO_FFMPEG)  # before the media folder is made: a refusal writes nothing anywhere
    media_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kit_replay-", dir=media_dir) as tmp:
        made = make_media(source, Path(tmp), fps=hz, max_video_bytes=max_video_bytes, ffmpeg=ffmpeg, ffprobe=ffprobe)
        # everything checked: write the task folder, then move the media into place
        csv_path, me_copy = task_dir / DEMO_FILE, task_dir / "kit_replay.py"
        replaced_csv = csv_path.is_file() and csv_path.read_bytes() != data_csv
        _write_atomic(csv_path, data_csv)
        if me_copy.resolve() != me:
            copied = "replaced an older copy" if me_copy.is_file() and me_copy.read_bytes() != data_me else (
                "unchanged" if me_copy.is_file() else "copied")
            if copied != "unchanged":
                _write_atomic(me_copy, data_me)
        else:
            copied = "this file itself"
        demo_mp4, setup_png = media_dir / "demo.mp4", media_dir / "setup.png"
        os.replace(made["demo"], demo_mp4)
        os.replace(made["setup"], setup_png)

    slug, module = task_dir.name, task_file.name
    task_json = _read_json(task_dir / "task.json")
    instruction = str((task_json.get("instruction") if isinstance(task_json, dict) else None)
                      or (entry or {}).get("instruction") or "")
    replay_cmd = ["inspect-robots run --embodiment yam_arena --policy yam_scripted \\",
                  f"  -E task=tasks/{slug}/{module} -E object_jitter=0 -E control_hz=20 -E cam_width=320 "
                  "-E cam_height=176 \\",
                  f"  --instruction {shlex.quote(instruction) if instruction else '<the instruction>'} "
                  f"--scorer task_score --max-steps {limit} \\",
                  f"  --store-frames --log-dir ~/yam-eval/{slug}-replay"]
    verify_cmd = f"python tasks/{slug}/kit_replay.py verify ~/yam-eval/{slug}-replay tasks/{slug}/{DEMO_FILE}"

    out(f"Trial {trial} (epoch {epoch}) of run {run}" + (f" (newest of {len(runs)} runs)" if len(runs) > 1 else ""))
    out(f"  score: {'unknown' if score is None else f'{score:g}'} ({score_from}); it ended: {ended}")
    out(f"  {summary['rows']} steps at {hz:g} Hz ({summary['duration_s']:.2f} s); every row passes the runner's "
        f"limits unchanged ({summary['on_limit']} values exactly on a per-step limit)")
    out(f"  scene: {scene_line}")
    out("")
    out(f"Task folder {task_dir}:")
    out(f"  {DEMO_FILE}: {'replaced' if replaced_csv else 'written'}, sha256 {sha}")
    out(f"  kit_replay.py: {copied}")
    out(f"  nothing else was written there ({words_note})")
    out(f"Media {media_dir} (upload these; never commit them):")
    out(f"  demo.mp4: {made['bytes'] / 1e6:.2f} MB, {made['width']}x{made['height']}, {made['how']}")
    out("  setup.png: the video's first frame")
    out("")
    for line in task_file_lines(task_file, sha):
        out(f"  {line}" if line else "")
    out("")
    out("Replay it once in Isaac with the official command (on the GPU machine, from the folder holding tasks/):")
    for line in replay_cmd:
        out(f"  {line}")
    out("Then check the replay (there, or here after copying its log folder back):")
    out(f"  {verify_cmd}")
    for note in notes:
        out(f"Note: {note}")
    return {"run": run, "trial": trial, "epoch": epoch, "score": score, "rows": rows, "sha256": sha,
            "csv": task_dir / DEMO_FILE, "kit_replay": task_dir / "kit_replay.py", "demo": demo_mp4,
            "setup": setup_png, "task_file": task_file, "run_limit": limit, "replay_command": "\n".join(replay_cmd),
            "verify_command": verify_cmd, "notes": notes, "scene": scene_line}


# -- verify -----------------------------------------------------------------------------------------------------------


def _verify_task(file: Path, task: str | os.PathLike | None) -> tuple[Path, Path]:
    """(task file, task folder) a replay is checked against: ``task`` (a task folder or the task file itself),
    else the folder the replayed file sits in."""
    if task is not None:
        p = Path(task).expanduser().resolve()
        if p.is_file() and p.suffix == ".py":
            return p, p.parent
        if not p.is_dir():
            raise PackError(f"--task {p}: no task folder or task file there")
        return find_task_file(p), p
    folder = file.expanduser().resolve().parent
    try:
        return find_task_file(folder), folder
    except PackError as exc:
        raise PackError(f"{exc}. verify checks a replay against the task file next to {file.name}: pass --task <the "
                        "task file or its folder> when that isn't where it sits") from None


def verify_run(log_dir: str | os.PathLike, file: str | os.PathLike, *, run: str | None = None,
               task: str | os.PathLike | None = None, control_hz: float | None = None, out: Any = print) -> bool:
    """Check a replay run against the file it replayed and the task it is for: every trial's executed actions
    equal the file's rows bit for bit (rows past its end equal its last row), no guard changed any action, the
    guards were the defaults, the run was of the task file as it is now (the eval log's environment_id names it and
    its environment_revision is the file's sha256) within the task's run limit (--max-steps and the steps taken),
    and task_score gave full marks. The task is ``task`` (a folder or its task file), else the task file next to
    ``file``. Prints what it finds; True when every trial passes."""
    log_dir = Path(log_dir).expanduser().resolve()
    rows, meta = load_actions(file, control_hz=control_hz)
    n = len(rows)
    task_file, task_dir = _verify_task(Path(file), task)
    now = file_sha256(task_file)[:12]
    limit, limit_from = run_limit_steps(task_dir)
    if limit is None:
        limit, limit_from = DEFAULT_RUN_LIMIT, f"{limit_from}; the template's {DEFAULT_RUN_LIMIT}"
    run, runs = _pick_run(log_dir, run)
    trials = _trials(log_dir, run)
    index = _eval_index(log_dir, run)
    identity = _run_identity(log_dir, run) or {}
    env_id, revision = identity.get("environment_id"), identity.get("environment_revision")
    ran = _task_of(env_id)
    out(f"{Path(file).name}: {n} rows at {meta['control_hz']:g} Hz, sha256 {meta['sha256']}")
    out(f"{task_file.name}: revision {now}; run limit {limit} steps ({limit_from})")
    out(f"run {run} in {log_dir}" + (f" (newest of {len(runs)})" if len(runs) > 1 else ""))
    if not trials:
        out("FAIL: the run has no trials")
        return False
    ok = True
    for (scene, epoch), path in sorted(trials.items(), key=lambda kv: kv[0][1]):
        problems: list[str] = []
        record = _record_json(log_dir, run, path.stem)
        entry = index.get((scene, epoch))
        rate = _trial_rate(record, identity)
        if rate is None:
            out(f"trial {epoch + 1}: FAIL: no record.json for it (records/{run}/{path.stem}/) and no eval log that "
                "gives the run's rate: copy the run's records folder back too")
            ok = False
            continue
        try:
            done, emeta = load_actions(path, control_hz=rate)
        except ValueError as exc:
            out(f"trial {epoch + 1}: FAIL: {exc}")
            ok = False
            continue
        s = len(done)
        if not math.isclose(emeta["control_hz"], meta["control_hz"]):
            problems.append(f"the run was at {emeta['control_hz']:g} Hz, the file at {meta['control_hz']:g} Hz")
        m = min(s, n)
        diff = np.argwhere(done[:m] != rows[:m])
        if len(diff):
            k, j = (int(x) for x in diff[0])
            problems.append(f"step {k} {COLUMNS[j]}: executed {_num(done[k, j])}, the file has {_num(rows[k, j])}")
        if s > n:
            tail = np.argwhere(done[n:] != rows[-1])
            if len(tail):
                k, j = (int(x) for x in tail[0])
                problems.append(f"step {n + k} {COLUMNS[j]}: executed {_num(done[n + k, j])} after the file ended, "
                                f"not its last row's {_num(rows[-1, j])}")
        kr = ((entry or {}).get("trial_metadata") or {}).get("kit_replay") or {}
        hold = kr.get("hold_steps")
        if isinstance(hold, int) and s > n + hold:
            problems.append(f"{s} steps: more than the file's {n} rows + {hold} hold steps (the stop didn't come)")
        if kr.get("sha256") and kr["sha256"] not in (meta["sha256"], meta["raw_sha256"]):  # an older copy: raw
            problems.append(f"the run replayed a file with sha256 {kr['sha256']}, not this one")
        if record is None:
            problems.append("no record.json (can't check the guards)")
            approvals = None
        else:
            approvals = sum(1 for e in record.get("events") or [] if (e or {}).get("kind") == "approval")
            if approvals:
                problems.append(f"the runner's guards changed {approvals} actions (approval events)")
            default, why = _guard_is_default(record)
            if not default:
                problems.append(f"not the official guards: {why}")
        # the run was of this task file as it is now, within its run limit
        if not env_id:
            problems.append("the run's logs don't say which task file it ran (no eval log or live log with an "
                            "environment_id)")
        elif ran != task_file.stem:
            problems.append(f"the run was of {f'task {ran}.py' if ran else 'no task file'} (environment_id {env_id}), "
                            f"not {task_file.name}: replay the task folder's file with the official command")
        elif not revision:
            problems.append(f"the run's log gives no environment_revision, so nothing shows it ran {task_file.name} "
                            f"as it is now (revision {now})")
        elif revision != now:
            problems.append(f"the run replayed {task_file.name} at revision {revision}, but the file is at {now} now: "
                            "it changed after the replay, so replay it again as it is")
        max_steps = identity.get("max_steps")
        if max_steps is None:
            max_steps = ((record or {}).get("configuration") or {}).get("max_steps")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool):
            problems.append(f"the run's --max-steps is unknown ({max_steps!r}); the official command uses "
                            f"--max-steps {limit}")
        elif max_steps > limit:
            problems.append(f"the run allowed {max_steps} steps (--max-steps), more than the task's run limit of "
                            f"{limit}: the official command uses --max-steps {limit}")
        if s > limit:
            problems.append(f"the trial took {s} steps, more than the task's run limit of {limit}")
        policy = (((record or {}).get("configuration") or {}).get("policy") or {}).get("name")
        reason = (entry or {}).get("termination") or ((record or {}).get("termination") or {}).get("reason")
        score = (entry or {}).get("score")
        if score is None:
            problems.append("no task_score for it (no eval log yet: the runner writes it when the run ends)")
        elif float(score) < FULL_MARKS:
            problems.append(f"task_score {float(score):g}, not full marks")
        ended = "" if s >= n else f"; it ended at step {s - 1}, before the file's last row ({n - 1})"
        line = (f"trial {epoch + 1}: {s} steps of --max-steps {max_steps} (run limit {limit}), task "
                f"{ran or '?'} revision {revision}, policy {policy}, ended {reason}, task_score "
                f"{'?' if score is None else f'{float(score):g}'}, guard changes {approvals}{ended}")
        if problems:
            ok = False
            out(f"{line}\n  FAIL: " + "\n  FAIL: ".join(problems))
        else:
            out(f"{line}\n  OK: every executed action is the file's row (then its last row), bit for bit")
    out("PASS" if ok else "FAIL")
    return ok


# -- command line -----------------------------------------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    rows, meta = load_actions(args.file, control_hz=args.control_hz)
    hz = meta["control_hz"]
    s = check_actions(rows, hz)
    hold = hold_steps(args.hold_s, hz)
    stop = stop_step(len(rows), hold)
    name = Path(args.file).name
    print(f"{name}: {s['rows']} rows at {hz:g} Hz ({s['duration_s']:.2f} s; rate from the {meta['rate_from']})")
    print(f"sha256 {meta['sha256']}")
    print(f"every row passes the runner's limits unchanged: {s['on_limit']} values exactly on a per-step limit; the "
          f"largest step is {100 * s['largest_step_fraction']:.1f}% of its limit (row {s['largest_step_at'][0]}, "
          f"{s['largest_step_at'][1]})")
    print(f"row 0 is {s['row0_from_ready']['left']:.4f} rad (left) and {s['row0_from_ready']['right']:.4f} rad "
          "(right) from the reset pose; grippers between "
          f"{s['grippers'][0]:.3f} and {s['grippers'][1]:.3f}")
    fits = stop + 1 <= args.max_steps
    print(f"with hold_s={args.hold_s:g} the replay asks to stop at step {stop}: {stop + 1} steps of --max-steps "
          f"{args.max_steps}" + ("" if fits else " (past the limit: fine only if the trial ends by success first)"))
    if meta["source"] == "csv":
        print("")
        print("task-file lines:")
        print("    import kit_replay  # noqa: E402  (after putting the task file's folder on sys.path)")
        print(f'    DEMO_SHA256 = "{meta["sha256"]}"')
        print("    def demo():")
        print(f'        return kit_replay.replay_demo("{name}", expect_sha256=DEMO_SHA256)')
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    ok = verify_run(args.log_dir, args.file, run=args.run, task=args.task, control_hz=args.control_hz)
    return 0 if ok else 1


def _cmd_pack(args: argparse.Namespace) -> int:
    pack_trial(args.log_dir, args.trial, args.task, args.media, any_score=args.any, other_task=args.other_task,
               over_limit=args.over_limit, scene_changed=args.scene_changed, run=args.run, hold_s=args.hold_s,
               blind_words_file=args.blind_words)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kit_replay.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="check a recording (CSV or action log) and print its sha256")
    p.add_argument("file")
    p.add_argument("--control-hz", type=float, default=None, help="the rate of an action log without record.json")
    p.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S)
    p.add_argument("--max-steps", type=int, default=DEFAULT_RUN_LIMIT)
    p.set_defaults(fn=_cmd_check)
    p = sub.add_parser("pack", help="turn a full-score trial into demo_actions.csv, demo.mp4 and setup.png")
    p.add_argument("log_dir", help="the run's --log-dir (copied back from the GPU machine)")
    p.add_argument("--trial", type=int, required=True, help="the page's trial number (from 1; epoch = N-1)")
    p.add_argument("--task", required=True, help="the task folder (gets demo_actions.csv and kit_replay.py)")
    p.add_argument("--media", default=None,
                   help="where demo.mp4 and setup.png go (default: robocurve/runs/<task folder>/upload in your home)")
    p.add_argument("--run", default=None, help="the run id (default: the newest run in the log folder)")
    p.add_argument("--any", action="store_true", help="pack it even if it didn't score full marks")
    p.add_argument("--other-task", action="store_true",
                   help="pack it even if it was recorded on a task file of another name and content")
    p.add_argument("--over-limit", action="store_true", help="pack it even if it is longer than the task's run limit")
    p.add_argument("--scene-changed", action="store_true",
                   help="pack it even if its scene stamp shows a library scene or scene files that differ from the "
                        "task folder's (then verify must PASS)")
    p.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S)
    p.add_argument("--blind-words", default=None, help="the blind-review word list (default: the kit's)")
    p.set_defaults(fn=_cmd_pack)
    p = sub.add_parser("verify", help="check a replay run executed the file exactly, ran this task file within its "
                                      "run limit, and scored full marks")
    p.add_argument("log_dir")
    p.add_argument("file")
    p.add_argument("--run", default=None, help="the run id (default: the newest run in the log folder)")
    p.add_argument("--task", default=None,
                   help="the task file the replay is for, or its folder (default: the task file next to FILE)")
    p.add_argument("--control-hz", type=float, default=None, help="the rate of an action log FILE without record.json")
    p.set_defaults(fn=_cmd_verify)
    args = ap.parse_args(argv)
    try:
        return int(args.fn(args))
    except (PackError, ValueError, OSError) as exc:
        print(f"kit_replay {args.command}: refused: {exc}", file=sys.stderr)
        return 1


install()

if __name__ == "__main__":
    raise SystemExit(main())
