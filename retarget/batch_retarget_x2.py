"""Batch SMPL-X (AMASS) -> AgiBot X2 retargeting with per-clip QC.

Walks AMASS subset directories, retargets every *_stageii.npz through GMR
(calibrated smplx_to_x2 config + posture-task patch), runs quality checks,
and keeps only passing clips as pickles mirroring the source layout. Sources
longer than the configured segment length are balanced into non-overlapping
parts before the SMPL-X forward pass, so long motions are retained without a
large per-worker memory spike.

QC per clip (kinematic + physics-adjacent):
  - NaN / non-finite output
  - joint velocity ceiling (motor sanity)
  - sole penetration below ground
  - stance-foot float above ground
  - self-collision fraction (MuJoCo collision query, sampled frames)
  - per-joint at-limit fractions and continuous unsafe-branch dwell

Usage (gmr env, from GMR repo root so its assets resolve):
  python ../Any2Any/retarget/batch_retarget_x2.py --subsets ACCAD --workers 12
  python ../Any2Any/retarget/batch_retarget_x2.py --subsets CMU BMLmovi --workers 24
"""

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import pickle
import sys
import time
import traceback
import zipfile

import numpy as np

try:
    from .retarget_quality import (
        ROOT_ANGULAR_VELOCITY_LIMIT,
        ROOT_LINEAR_VELOCITY_LIMIT,
        X2_ARM_COLLISION_MARGIN,
        foot_collision_geoms,
        joint_limit_report,
        limit_joint_velocity,
        root_motion_metrics,
        sole_height,
    )
except ImportError:  # direct script execution
    from retarget_quality import (
        ROOT_ANGULAR_VELOCITY_LIMIT,
        ROOT_LINEAR_VELOCITY_LIMIT,
        X2_ARM_COLLISION_MARGIN,
        foot_collision_geoms,
        joint_limit_report,
        limit_joint_velocity,
        root_motion_metrics,
        sole_height,
    )

HERE = pathlib.Path(__file__).resolve().parent
ANY2ANY_ROOT = HERE.parent
SR_ROOT = ANY2ANY_ROOT.parent              # Sonic_Retarget/
GMR_ROOT = SR_ROOT / "GMR"
AMASS_ROOT = pathlib.Path(os.environ.get(
    "X2_AMASS_ROOT", "/data/motion/AMASS/SMPL-X_N",
))
OUT_ROOT = ANY2ANY_ROOT / "retargeted_dataset"
X2_XML = GMR_ROOT / "assets" / "agibot_x2" / "x2_mocap.xml"

MIN_SECONDS = 2.0
MAX_SECONDS = 40.0
VEL_LIMIT = 15.0          # rad/s, fail above this
PENETRATION_LIMIT = -0.06  # m sole below ground
FLOAT_LIMIT = 0.06         # m median stance sole height
SELFCOL_LIMIT = 0.10       # fraction of sampled frames with self-collision

_worker = {}


def _tpose_height(model, betas, torch):
    """Stature of the T-posed SMPL-X body (vertex z-range; model frame is Y-up)."""
    out = model(betas=betas, global_orient=torch.zeros(1, 3), body_pose=torch.zeros(1, 63),
                transl=torch.zeros(1, 3), left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45), jaw_pose=torch.zeros(1, 3),
                leye_pose=torch.zeros(1, 3), reye_pose=torch.zeros(1, 3))
    v = out.vertices[0].detach().numpy()
    return float(v[:, 1].max() - v[:, 1].min())


def _load_segment_arrays(path, start_frame, end_frame):
    """Load only the SMPL-X parameters needed for one source-frame segment."""
    with np.load(path, allow_pickle=True) as raw:
        d = {
            key: raw[key]
            for key in (
                "gender", "betas", "pose_body", "root_orient", "trans",
                "mocap_frame_rate",
            )
        }
    frame_slice = slice(start_frame, end_frame)
    for key in ("pose_body", "root_orient", "trans"):
        d[key] = np.asarray(d[key][frame_slice])
    return d


def _load_canonical(path, start_frame=None, end_frame=None):
    """Shape-neutralized SMPL-X load (same return contract as load_smplx_file).

    Replaces the subject's betas with the canonical neutral zero-beta skeleton
    and scales the root trajectory by the stature ratio, so identical poses
    from different performers produce bit-identical retarget inputs. This kills
    GMR's h^2 subject-height dependence (short subjects -> squatting robot):
    verified 2026-07-16, knee angle spread across 1.53-1.95m synthetic bodies
    drops from +/-40 deg (stock GMR) to exactly 0.
    """
    import numpy as np
    import smplx
    torch = _worker["torch"]
    canon = _worker["canon"]
    d = _load_segment_arrays(path, start_frame, end_frame)
    g = str(d["gender"])
    sub = _worker["subj_models"].get(g)
    if sub is None:
        sub = smplx.create("assets/body_models", "smplx", gender=g, use_pca=False)
        _worker["subj_models"][g] = sub
    h_s = _tpose_height(sub, torch.tensor(d["betas"]).float().view(1, -1), torch)
    d["betas"] = np.zeros(16, dtype=np.float32)
    d["trans"] = d["trans"].astype(np.float32) * (_worker["canon_h"] / h_s)
    n = d["pose_body"].shape[0]
    out = canon(betas=torch.zeros(1, 16),
                global_orient=torch.tensor(d["root_orient"]).float(),
                body_pose=torch.tensor(d["pose_body"]).float(),
                transl=torch.tensor(d["trans"]).float(),
                left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45),
                jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3),
                reye_pose=torch.zeros(n, 3), return_full_pose=True)
    # constant for every clip: calibrated table ratio, NOT a per-subject value
    h_arg = _worker["canonical_height"]
    return d, canon, out, h_arg


def _load_subject_segment(path, start_frame=None, end_frame=None):
    """Legacy subject-shaped loader with pre-forward source-frame slicing."""
    torch = _worker["torch"]
    d = _load_segment_arrays(path, start_frame, end_frame)
    gender = str(d["gender"])
    body_model = _worker["subj_models"].get(gender)
    if body_model is None:
        import smplx
        body_model = smplx.create(
            "assets/body_models", "smplx", gender=gender, use_pca=False,
        )
        _worker["subj_models"][gender] = body_model
    n = d["pose_body"].shape[0]
    betas = torch.tensor(d["betas"]).float().view(1, -1)
    out = body_model(
        betas=betas,
        global_orient=torch.tensor(d["root_orient"]).float(),
        body_pose=torch.tensor(d["pose_body"]).float(),
        transl=torch.tensor(d["trans"]).float(),
        left_hand_pose=torch.zeros(n, 45), right_hand_pose=torch.zeros(n, 45),
        jaw_pose=torch.zeros(n, 3), leye_pose=torch.zeros(n, 3),
        reye_pose=torch.zeros(n, 3), return_full_pose=True,
    )
    beta0 = float(np.asarray(d["betas"]).reshape(-1)[0])
    return d, body_model, out, 1.66 + 0.1 * beta0


def worker_init(canonical=True, canonical_height=1.8):
    sys.path.insert(0, str(GMR_ROOT))
    sys.path.insert(0, str(HERE))
    os.chdir(GMR_ROOT)  # GMR resolves assets relative to its root
    import torch
    torch.set_num_threads(1)        # workers each spawning 32 threads thrashes
    torch.set_grad_enabled(False)   # smplx forward needs no autograd: ~2x memory
    import mujoco
    from native_gmr import NativeGMR  # noqa
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast  # noqa
    from mink.limits import CollisionAvoidanceLimit  # noqa
    _worker["mujoco"] = mujoco
    _worker["GMR"] = NativeGMR
    _worker["CollisionAvoidanceLimit"] = CollisionAvoidanceLimit
    _worker["frames"] = get_smplx_data_offline_fast
    _worker["canonical"] = bool(canonical)
    _worker["canonical_height"] = float(canonical_height)
    _worker["torch"] = torch
    _worker["subj_models"] = {}
    if canonical:
        import smplx
        _worker["canon"] = smplx.create("assets/body_models", "smplx",
                                        gender="neutral", use_pca=False)
        _worker["canon_h"] = _tpose_height(_worker["canon"], torch.zeros(1, 16), torch)
    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    _worker["model"], _worker["data"] = model, data
    _worker["foot_geoms"] = foot_collision_geoms(model, mujoco)
    # Whitelist: near-neighbor bodies on the kinematic tree (distance <= 2).
    # MuJoCo only auto-excludes direct parent-child; grandparent shells like
    # pelvis<->hip_roll permanently nest inside each other on the X2.
    def ancestors(b):
        chain = []
        while b > 0:
            chain.append(b)
            b = model.body_parentid[b]
        chain.append(0)
        return chain
    base = set()
    chains = {b: ancestors(b) for b in range(model.nbody)}
    for b1 in range(model.nbody):
        for b2 in range(b1 + 1, model.nbody):
            c1, c2 = chains[b1], chains[b2]
            common = next(a for a in c1 if a in c2)
            dist = c1.index(common) + c2.index(common)
            if dist <= 2:
                base.add((b1, b2))
    _worker["contact_baseline"] = base
    # silence GMR's per-init stdout spam
    sys.stdout = open(os.devnull, "w")


def qc_checks(qpos, fps):
    mujoco = _worker["mujoco"]
    model, data = _worker["model"], _worker["data"]
    root_pos, root_rot, dof = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
    rep = {}

    rep["finite"] = bool(np.isfinite(qpos).all())
    if not rep["finite"]:
        rep["fail"] = "nan"
        return rep

    velocity_frames = dof[5:] if len(dof) > 6 else dof
    vel = (float(np.abs(np.diff(velocity_frames, axis=0)).max() * fps)
           if len(velocity_frames) > 1 else 0.0)  # skip IK settle-in
    rep["max_joint_vel"] = float(vel)

    rep["max_root_vel"], rep["max_root_ang_vel"] = root_motion_metrics(qpos, fps)
    rep["at_limit"], unexpected_limits = joint_limit_report(
        model, mujoco, dof, fps=fps,
    )
    rep["unexpected_at_limit"] = unexpected_limits

    foot_geoms = _worker["foot_geoms"]
    baseline = _worker["contact_baseline"]
    soles, selfcol = [], 0
    sample = range(0, len(qpos), 5)
    for t in sample:
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root_pos[t], root_rot[t], dof[t]
        mujoco.mj_forward(model, data)
        soles.append(sole_height(model, data, foot_geoms))
        # self-collision: deep penetration between non-neighbor body pairs
        # (convex-hull grazing of a few mm, e.g. hand brushing thigh, is benign)
        for c in data.contact[: data.ncon]:
            if c.dist < -0.015:
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if (min(b1, b2), max(b1, b2)) not in baseline:
                    selfcol += 1
                    break
    soles = np.array(soles)
    rep["sole_min"] = float(soles.min())
    rep["sole_stance_med"] = float(np.percentile(soles, 20))
    rep["selfcol_frac"] = round(selfcol / len(list(sample)), 3)
    rep["root_med"] = float(np.median(root_pos[:, 2]))
    grounded_gait = rep["root_med"] > 0.45  # crawls/lies legitimately fly feet

    if vel > VEL_LIMIT:
        rep["fail"] = "velocity"
    elif rep["max_root_vel"] > ROOT_LINEAR_VELOCITY_LIMIT:
        rep["fail"] = "root_velocity"
    elif rep["max_root_ang_vel"] > ROOT_ANGULAR_VELOCITY_LIMIT:
        rep["fail"] = "root_angular_velocity"
    elif unexpected_limits:
        rep["fail"] = "joint_limit"
    elif rep["sole_min"] < PENETRATION_LIMIT:
        rep["fail"] = "penetration"
    elif grounded_gait and rep["sole_stance_med"] > FLOAT_LIMIT:
        rep["fail"] = "float"
    elif rep["selfcol_frac"] > SELFCOL_LIMIT:
        rep["fail"] = "self_collision"
    else:
        rep["fail"] = None
    return rep


def clip_frame_info(src):
    """Read frame count and FPS without decompressing the large pose array."""
    try:
        with zipfile.ZipFile(src) as archive:
            with archive.open("pose_body.npy") as pose_file:
                version = np.lib.format.read_magic(pose_file)
                shape, _, _ = np.lib.format._read_array_header(pose_file, version)
            if "mocap_frame_rate.npy" in archive.namelist():
                with archive.open("mocap_frame_rate.npy") as fps_file:
                    fps = float(np.lib.format.read_array(fps_file, allow_pickle=True))
            else:
                fps = 120.0
        if not shape or shape[0] < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid clip metadata: shape={shape}, fps={fps}")
        return int(shape[0]), fps
    except Exception:
        with np.load(src, allow_pickle=True) as z:
            n = int(z["pose_body"].shape[0])
            fps = float(z["mocap_frame_rate"]) if "mocap_frame_rate" in z else 120.0
        return n, fps


def segment_frame_ranges(num_frames, fps, max_seconds):
    """Partition a clip into balanced, non-overlapping, bounded segments.

    Balancing avoids a tiny final remainder: for example, 81 seconds becomes
    three 27-second clips instead of 40 + 40 + 1 (which would fail MIN_SECONDS).
    """
    if num_frames < 1 or fps <= 0 or max_seconds <= 0:
        raise ValueError(
            f"invalid segmentation inputs: frames={num_frames}, "
            f"fps={fps}, max_seconds={max_seconds}"
        )
    max_frames = max(1, int(np.floor(float(fps) * max_seconds + 1e-9)))
    part_count = int(np.ceil(num_frames / max_frames))
    boundaries = np.rint(np.linspace(0, num_frames, part_count + 1)).astype(int)
    return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]


def segment_output_path(out_path, part_index, part_count):
    if part_count == 1:
        return out_path
    return out_path.with_name(
        f"{out_path.stem}__part{part_index:03d}{out_path.suffix}"
    )


def recipe_locomotion_paths(manifest_path, splits=("train", "val", "test")):
    """Return AMASS-relative locomotion sources selected by the recipe manifest."""
    allowed_splits = set(splits)
    paths = []
    with open(manifest_path, encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if (
                entry.get("category") == "locomotion"
                and entry.get("split") in allowed_splits
            ):
                paths.append(pathlib.Path(entry["path"]))
    return paths


def clip_tasks(rel):
    """Expand one source path into independently retargetable segment tasks."""
    src = AMASS_ROOT / rel
    try:
        num_frames, source_fps = clip_frame_info(src)
        ranges = segment_frame_ranges(num_frames, source_fps, MAX_SECONDS)
    except Exception:
        # Preserve the normal error-reporting path for corrupt/missing inputs.
        num_frames, source_fps, ranges = None, None, [(None, None)]
    base_out = OUT_ROOT / rel.with_suffix(".pkl")
    part_count = len(ranges)
    tasks = []
    for part_index, (start_frame, end_frame) in enumerate(ranges):
        out_path = segment_output_path(base_out, part_index, part_count)
        clip_id = str(rel) if part_count == 1 else f"{rel}#part{part_index:03d}"
        tasks.append((
            rel, start_frame, end_frame, out_path, clip_id,
            part_index, part_count, source_fps,
        ))
    return tasks


def ground_snap(qpos):
    """Shift root z so the 5th-percentile sole height sits on the ground.

    AMASS subsets differ in floor calibration (BMLmovi/CMU sources often sit
    below z=0); the IK faithfully reproduces that as sole penetration. A rigid
    vertical shift is the standard motion-lib correction and does not alter
    the motion itself. Returns (qpos, applied_offset_m).
    """
    mujoco = _worker["mujoco"]
    model, data = _worker["model"], _worker["data"]
    foot_geoms = _worker["foot_geoms"]
    soles = []
    for t in range(0, len(qpos), 5):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        soles.append(sole_height(model, data, foot_geoms))
    off = float(np.percentile(np.array(soles), 5))
    if abs(off) > 0.005:
        qpos = qpos.copy()
        qpos[:, 2] -= off
    return qpos, off


def ground_track(qpos, fps, contact_band=0.03, max_rate=0.6, smooth_s=0.15):
    """Per-frame ground tracking: keep the stance foot ON the floor.

    ``ground_snap`` shifts the whole clip by one constant (the 5th-percentile
    sole height).  The solved sole height swings ~5 cm within a single walk,
    so one constant cannot suit every frame: pinning the deepest frames
    leaves the rest hanging (measured on ACCAD B3 walk: median sole +29 mm
    after the snap, 47% of frames more than 3 cm off the floor, only 9.6%
    actually touching).

    Track it instead.  Per frame, take the lowest sole; a frame whose sole is
    within ``contact_band`` of the running floor is a stance frame and its
    sole is driven to zero.  Flight frames keep the offset from the last
    stance frame, so a jump stays a jump.  The correction is rate-limited and
    smoothed so it never injects vertical velocity the motion did not have.

    Returns (qpos, mean_offset_m).
    """
    mujoco = _worker["mujoco"]
    model, data = _worker["model"], _worker["data"]
    foot_geoms = _worker["foot_geoms"]

    soles = np.empty(len(qpos))
    for t in range(len(qpos)):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        soles[t] = sole_height(model, data, foot_geoms)

    floor = float(np.percentile(soles, 5))
    target = np.empty(len(qpos))
    hold = soles[0] - floor
    for t in range(len(qpos)):
        rel = soles[t] - floor
        if rel <= contact_band:          # stance: put this sole on the floor
            hold = rel
        target[t] = floor + hold         # flight: keep the last stance offset

    # rate limit, then smooth: a correction must not add vertical velocity
    limit = max_rate / fps
    limited = np.empty_like(target)
    limited[0] = target[0]
    for t in range(1, len(target)):
        step = np.clip(target[t] - limited[t - 1], -limit, limit)
        limited[t] = limited[t - 1] + step
    window = max(1, int(round(smooth_s * fps)) | 1)
    kernel = np.ones(window) / window
    padded = np.pad(limited, window // 2, mode="edge")
    offset = np.convolve(padded, kernel, mode="valid")[: len(qpos)]

    qpos = qpos.copy()
    qpos[:, 2] -= offset
    return qpos, float(offset.mean())


GROUND_SNAP = False   # set by main() via --ground_snap
GROUND_TRACK = True   # per-frame stance tracking; --constant_ground falls back


def retarget_frames(frames, fps, human_height):
    """Retarget an in-memory human trajectory with the production X2 stack."""
    rt = _worker["GMR"](
        actual_human_height=human_height, src_human="smplx", tgt_robot="agibot_x2",
        verbose=False,
    )
    # Arm-vs-lower-body + hand-vs-hand collision avoidance inside the IK QP.
    mujoco = _worker["mujoco"]
    model = rt.model

    def body_geoms(names):
        geoms = []
        for name in names:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            geoms += [
                geom for geom in range(model.ngeom)
                if model.geom_bodyid[geom] == body and model.geom_contype[geom] != 0
            ]
        return geoms

    arm_l = [f"left_{part}_link" for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]
    arm_r = [f"right_{part}_link" for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]
    lower = ["pelvis", "torso_link"] + [
        f"{side}_{part}_link"
        for side in ("left", "right")
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ]
    geoms_l, geoms_r, geoms_lower = body_geoms(arm_l), body_geoms(arm_r), body_geoms(lower)
    rt.ik_limits.append(_worker["CollisionAvoidanceLimit"](
        model, [(geoms_l + geoms_r, geoms_lower), (geoms_l, geoms_r)],
        minimum_distance_from_collisions=X2_ARM_COLLISION_MARGIN,
    ))
    for _ in range(20):
        rt.retarget(frames[0])

    qpos, previous = [], None
    for frame in frames:
        current = rt.retarget(frame).copy()
        # No low-pass on the reference: a constant-gain filter throttles fast
        # transients (punches lose ~2/3 of their joint velocity).  Keep only
        # the per-frame velocity clamp as a safety net against IK jumps.
        current, clipped = limit_joint_velocity(previous, current, fps)
        if clipped:
            rt.configuration.update(current)
        qpos.append(current.copy())
        previous = current
    return np.stack(qpos)


def process_clip(args):
    (rel, start_frame, end_frame, out_path, clip_id,
     part_index, part_count, source_fps) = args
    src = AMASS_ROOT / rel
    t0 = time.time()
    try:
        if source_fps is None:
            _, source_fps = clip_frame_info(src)
        dur0 = (end_frame - start_frame) / source_fps
        if dur0 < MIN_SECONDS:
            return {"clip": clip_id, "source_clip": str(rel),
                    "fail": "duration", "seconds": round(dur0, 1)}
        if _worker["canonical"]:
            smplx_data, bm, out, h = _load_canonical(
                str(src), start_frame, end_frame,
            )
        else:
            smplx_data, bm, out, h = _load_subject_segment(
                str(src), start_frame, end_frame,
            )
        frames, fps = _worker["frames"](smplx_data, bm, out, tgt_fps=30)
        dur = len(frames) / fps
        if dur < MIN_SECONDS:
            return {"clip": clip_id, "source_clip": str(rel),
                    "fail": "duration", "seconds": round(dur, 1)}
        qpos = retarget_frames(frames, fps, h)
        snap_off = 0.0
        if GROUND_TRACK:
            qpos, snap_off = ground_track(qpos, fps)
        elif GROUND_SNAP:
            qpos, snap_off = ground_snap(qpos)
        del smplx_data, bm, out, frames
        import gc
        gc.collect()
        rep = qc_checks(qpos, fps)
        rep.update({"clip": clip_id, "source_clip": str(rel),
                    "part": part_index, "parts": part_count,
                    "source_frames": [start_frame, end_frame],
                    "seconds": round(dur, 1),
                    "ik_seconds": round(time.time() - t0, 1)})
        if GROUND_TRACK or GROUND_SNAP:
            rep["ground_offset_m"] = round(snap_off, 4)
            rep["ground_mode"] = "track" if GROUND_TRACK else "snap"
        if rep["fail"] is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump({"robot": "agibot_x2", "fps": fps,
                             "root_pos": qpos[:, :3], "root_rot": qpos[:, 3:7],
                             "dof_pos": qpos[:, 7:], "source": str(rel),
                             "source_frame_start": start_frame,
                             "source_frame_end": end_frame,
                             "source_part": part_index,
                             "source_parts": part_count}, f)
        return rep
    except Exception as e:
        return {"clip": clip_id, "source_clip": str(rel), "fail": "error",
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-400:]}


def main():
    global OUT_ROOT, GROUND_SNAP, GROUND_TRACK, MAX_SECONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=[])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--skip_judged", action="store_true",
                    help="also skip clips already judged in qc_report.jsonl "
                         "(default only skips clips with an output pkl, so "
                         "prior FAILs re-run on every resume)")
    ap.add_argument("--ground_snap", dest="ground_snap", action="store_true",
                    help="enable clip-level ground snap (default)")
    ap.add_argument("--no_ground_snap", dest="ground_snap", action="store_false",
                    help="disable clip-level ground snap")
    ap.set_defaults(ground_snap=True)
    ap.add_argument("--constant_ground", dest="ground_track", action="store_false",
                    help="use the old clip-constant ground snap instead of "
                         "per-frame stance tracking")
    ap.set_defaults(ground_track=True)
    ap.add_argument("--only_failed", default=None, metavar="REASON",
                    help="re-run only clips whose LATEST qc_report entry failed "
                         "with this reason (e.g. penetration); implies not skipping them")
    ap.add_argument("--max_seconds", type=float, default=MAX_SECONDS,
                    help="maximum segment length (default 40); longer source "
                         "motions are split into balanced non-overlapping parts")
    ap.add_argument("--clip_list", default=None,
                    help="file of AMASS-relative npz paths (one per line); "
                         "overrides --subsets scanning")
    ap.add_argument("--recipe_manifest", default=None,
                    help="recipe JSONL; retarget only category=locomotion")
    ap.add_argument("--recipe_splits", nargs="+", default=["train", "val", "test"],
                    help="splits selected from --recipe_manifest")
    ap.add_argument("--out_root", default=None,
                    help="output dir override (keeps the original "
                         "retargeted_dataset untouched for parallel versions)")
    ap.add_argument("--subject_shape", action="store_true",
                    help="preserve each AMASS subject's SMPL-X betas (legacy; "
                         "default uses a canonical neutral body to avoid the "
                         "known subject-height-squared distortion)")
    ap.add_argument("--canonical_height", type=float,
                    default=float(os.environ.get("ANY2ANY_CANON_HEIGHT", "1.8")),
                    help="constant height passed to the calibrated GMR table "
                         "in canonical mode (default 1.8 m)")
    args = ap.parse_args()

    if args.clip_list and args.recipe_manifest:
        ap.error("--clip_list and --recipe_manifest are mutually exclusive")
    if not AMASS_ROOT.is_dir():
        ap.error(f"AMASS source root does not exist: {AMASS_ROOT}")

    if args.out_root:
        # Workers chdir into GMR_ROOT while loading the retargeter.  Resolve the
        # override here so their output paths do not become relative to GMR.
        OUT_ROOT = pathlib.Path(args.out_root).resolve()
    GROUND_SNAP = args.ground_snap
    GROUND_TRACK = args.ground_track
    MAX_SECONDS = args.max_seconds

    rerun_set = None
    if args.only_failed:
        latest = {}
        report = OUT_ROOT / "qc_report.jsonl"
        if report.exists():
            for line in open(report):
                try:
                    r = json.loads(line)
                    latest[r["clip"]] = r
                except Exception:
                    pass
        rerun_set = {c for c, r in latest.items() if r.get("fail") == args.only_failed}
        print(f"only_failed={args.only_failed}: {len(rerun_set)} clips to re-run")

    judged = set()
    if args.skip_judged:
        report = OUT_ROOT / "qc_report.jsonl"
        if report.exists():
            for line in open(report):
                try:
                    judged.add(json.loads(line)["clip"])
                except Exception:
                    pass
        print(f"skip_judged: {len(judged)} clips already in report")

    source_clips = []
    if args.clip_list:
        listed = [pathlib.Path(l.strip()) for l in open(args.clip_list) if l.strip()]
        print(f"clip_list: {len(listed)} entries")
        source_clips.extend(listed)
    elif args.recipe_manifest:
        listed = recipe_locomotion_paths(args.recipe_manifest, args.recipe_splits)
        print(f"recipe locomotion: {len(listed)} entries")
        source_clips.extend(listed)
    for sub in (args.subsets if not (args.clip_list or args.recipe_manifest) else []):
        root = AMASS_ROOT / sub
        for p in sorted(root.rglob("*_stageii.npz")):
            if "stagei" in p.name and "stageii" not in p.name:
                continue
            source_clips.append(p.relative_to(AMASS_ROOT))
    if args.max_clips:
        source_clips = source_clips[: args.max_clips]

    clips = []
    segmented_sources = 0
    for rel in source_clips:
        tasks = clip_tasks(rel)
        if len(tasks) > 1:
            segmented_sources += 1
        for task in tasks:
            clip_id, out_path = task[4], task[3]
            if rerun_set is not None:
                # A legacy unsplit duration failure is keyed by source path;
                # a new failure is keyed by its individual #partNNN id.
                if clip_id in rerun_set or str(rel) in rerun_set:
                    clips.append(task)
                continue
            if clip_id in judged:
                continue
            if not out_path.exists():
                clips.append(task)
    print(f"{len(clips)} segments to retarget with {args.workers} workers "
          f"({segmented_sources} long sources split)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUT_ROOT / "qc_report.jsonl"
    n_pass = n_fail = 0
    t0 = time.time()
    with mp.Pool(args.workers, initializer=worker_init,
                 initargs=(not args.subject_shape, args.canonical_height),
                 maxtasksperchild=25) as pool, \
            open(report_path, "a") as rep_f:
        for i, rep in enumerate(pool.imap_unordered(process_clip, clips, chunksize=1)):
            rep.pop("trace", None) if rep.get("fail") != "error" else None
            rep_f.write(json.dumps(rep) + "\n")
            rep_f.flush()
            if rep.get("fail") is None:
                n_pass += 1
            else:
                n_fail += 1
            if (i + 1) % 20 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(clips) - i - 1) / rate / 60
                print(f"[{i+1}/{len(clips)}] pass {n_pass} fail {n_fail} "
                      f"({rate*60:.0f} clips/min, ETA {eta:.0f} min)", flush=True)
    print(f"done: {n_pass} passed, {n_fail} failed -> {OUT_ROOT}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
