"""Batch OMOMO (SMPL-X npz, test split) -> AgiBot X2 for the Any2Any eval set.

Same quality stack as batch_retarget_lafan_x2.py: IK-QP collision avoidance
(arms vs lower body + hand vs hand), trajectory rate limiter with warm-start
feedback, ground snap, and the AMASS QC battery. Paper Sec 4.1.1 uses "500
selected clips"; we retarget the OMOMO test split (602 seqs) and the eval
harness takes the QC-passing set (capped at 500, sorted for determinism).

Usage (gmr env):  python ../Any2Any/retarget/batch_retarget_omomo_x2.py --workers 6
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

import numpy as np

try:
    from .retarget_quality import (
        ROOT_ANGULAR_VELOCITY_LIMIT,
        ROOT_LINEAR_VELOCITY_LIMIT,
        X2_ARM_COLLISION_MARGIN,
        foot_collision_geoms,
        joint_limit_report,
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
        root_motion_metrics,
        sole_height,
    )

HERE = pathlib.Path(__file__).resolve().parent
SR_ROOT = HERE.parent.parent
GMR_ROOT = SR_ROOT / "GMR"
SRC_ROOT = SR_ROOT / "eval_data" / "omomo_smplx"
OUT_ROOT = SR_ROOT / "eval_data" / "omomo_x2"
X2_XML = GMR_ROOT / "assets" / "agibot_x2" / "x2_mocap.xml"

VEL_LIMIT = 15.0
PENETRATION_LIMIT = -0.06
FLOAT_LIMIT = 0.06
SELFCOL_LIMIT = 0.10
MIN_SECONDS = 2.0

_worker = {}


def _tpose_height(model, betas, torch):
    out = model(
        betas=betas, global_orient=torch.zeros(1, 3), body_pose=torch.zeros(1, 63),
        transl=torch.zeros(1, 3), left_hand_pose=torch.zeros(1, 45),
        right_hand_pose=torch.zeros(1, 45), jaw_pose=torch.zeros(1, 3),
        leye_pose=torch.zeros(1, 3), reye_pose=torch.zeros(1, 3),
    )
    vertices = out.vertices[0].detach().numpy()
    return float(vertices[:, 1].max() - vertices[:, 1].min())


def _load_canonical(path):
    import smplx

    torch = _worker["torch"]
    with np.load(path, allow_pickle=True) as raw:
        source = {key: raw[key] for key in raw.files}
    gender = str(source["gender"])
    subject_model = _worker["subject_models"].get(gender)
    if subject_model is None:
        subject_model = smplx.create(
            "assets/body_models", "smplx", gender=gender, use_pca=False,
        )
        _worker["subject_models"][gender] = subject_model
    source_height = _tpose_height(
        subject_model, torch.tensor(source["betas"]).float().view(1, -1), torch,
    )
    source["betas"] = np.zeros(16, dtype=np.float32)
    source["trans"] = source["trans"].astype(np.float32) * (
        _worker["canonical_body_height"] / source_height
    )
    frame_count = source["pose_body"].shape[0]
    canonical = _worker["canonical_model"]
    output = canonical(
        betas=torch.zeros(1, 16),
        global_orient=torch.tensor(source["root_orient"]).float(),
        body_pose=torch.tensor(source["pose_body"]).float(),
        transl=torch.tensor(source["trans"]).float(),
        left_hand_pose=torch.zeros(frame_count, 45),
        right_hand_pose=torch.zeros(frame_count, 45),
        jaw_pose=torch.zeros(frame_count, 3),
        leye_pose=torch.zeros(frame_count, 3),
        reye_pose=torch.zeros(frame_count, 3),
        return_full_pose=True,
    )
    return source, canonical, output, _worker["canonical_height"]


def worker_init(canonical=True, canonical_height=1.8):
    sys.path.insert(0, str(GMR_ROOT))
    os.chdir(GMR_ROOT)
    import torch
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa
    from general_motion_retargeting.utils.smpl import (  # noqa
        get_smplx_data_offline_fast, load_smplx_file,
    )
    from mink.limits import CollisionAvoidanceLimit  # noqa
    _worker["mujoco"] = mujoco
    _worker["GMR"] = GMR
    _worker["load"] = load_smplx_file
    _worker["frames"] = get_smplx_data_offline_fast
    _worker["CollisionAvoidanceLimit"] = CollisionAvoidanceLimit
    _worker["canonical"] = bool(canonical)
    _worker["canonical_height"] = float(canonical_height)
    if canonical:
        import smplx

        _worker["torch"] = torch
        _worker["canonical_model"] = smplx.create(
            "assets/body_models", "smplx", gender="neutral", use_pca=False,
        )
        _worker["subject_models"] = {}
        _worker["canonical_body_height"] = _tpose_height(
            _worker["canonical_model"], torch.zeros(1, 16), torch,
        )
    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    _worker["model"], _worker["data"] = model, data
    _worker["foot_geoms"] = foot_collision_geoms(model, mujoco)
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
            if c1.index(common) + c2.index(common) <= 2:
                base.add((b1, b2))
    _worker["contact_baseline"] = base
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
           if len(velocity_frames) > 1 else 0.0)
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
    grounded_gait = rep["root_med"] > 0.45
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


def ground_snap(qpos):
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


def process_clip(args):
    rel, out_path = args
    src = SRC_ROOT / rel
    t0 = time.time()
    try:
        if _worker["canonical"]:
            smplx_data, bm, out, h = _load_canonical(str(src))
        else:
            smplx_data, bm, out, h = _worker["load"](str(src), "assets/body_models")
        frames, fps = _worker["frames"](smplx_data, bm, out, tgt_fps=30)
        dur = len(frames) / fps
        if dur < MIN_SECONDS:
            return {"clip": str(rel), "fail": "duration", "seconds": round(dur, 1)}
        rt = _worker["GMR"](actual_human_height=h, src_human="smplx",
                            tgt_robot="agibot_x2", verbose=False)
        mujoco = _worker["mujoco"]
        model = rt.model
        def body_geoms(names):
            outg = []
            for n in names:
                b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                outg += [g for g in range(model.ngeom)
                         if model.geom_bodyid[g] == b and model.geom_contype[g] != 0]
            return outg
        arm_l = [f"left_{p}_link" for p in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]
        arm_r = [f"right_{p}_link" for p in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]
        low = ["pelvis", "torso_link"] + [f"{s}_{p}_link" for s in ("left", "right")
                                          for p in ("hip_pitch", "hip_roll", "hip_yaw", "knee")]
        ga_l, ga_r, gl = body_geoms(arm_l), body_geoms(arm_r), body_geoms(low)
        rt.ik_limits.append(_worker["CollisionAvoidanceLimit"](
            model, [(ga_l + ga_r, gl), (ga_l, ga_r)],
            minimum_distance_from_collisions=X2_ARM_COLLISION_MARGIN))
        for _ in range(20):
            rt.retarget(frames[0])
        VMAX = 12.0
        step = VMAX / fps
        qs, prev = [], None
        for f in frames:
            q = rt.retarget(f).copy()
            if prev is not None and np.abs(q[7:] - prev[7:]).max() > step:
                q[7:] = prev[7:] + np.clip(q[7:] - prev[7:], -step, step)
                rt.configuration.update(q)
            qs.append(q.copy())
            prev = q
        qpos = np.stack(qs)
        qpos, snap_off = ground_snap(qpos)
        del smplx_data, bm, out, frames, rt
        import gc
        gc.collect()
        rep = qc_checks(qpos, fps)
        rep.update({"clip": str(rel), "seconds": round(len(qpos) / fps, 1),
                    "ik_seconds": round(time.time() - t0, 1),
                    "ground_snap_m": round(snap_off, 4)})
        if rep["fail"] is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump({"robot": "agibot_x2", "fps": fps,
                             "root_pos": qpos[:, :3], "root_rot": qpos[:, 3:7],
                             "dof_pos": qpos[:, 7:], "source": str(rel)}, f)
        return rep
    except Exception as e:
        return {"clip": str(rel), "fail": "error",
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-400:]}


def main():
    global OUT_ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--out_root", type=pathlib.Path, default=OUT_ROOT,
                    help="write to a separate directory; existing clips are skipped")
    ap.add_argument("--subject_shape", action="store_true",
                    help="preserve per-subject SMPL-X shape (legacy)")
    ap.add_argument("--canonical_height", type=float,
                    default=float(os.environ.get("ANY2ANY_CANON_HEIGHT", "1.8")))
    args = ap.parse_args()

    OUT_ROOT = args.out_root

    pattern = "*.npz" if args.split == "all" else f"{args.split}__*.npz"
    clips = []
    for p in sorted(SRC_ROOT.glob(pattern)):
        rel = p.relative_to(SRC_ROOT)
        out_path = OUT_ROOT / rel.with_suffix(".pkl")
        if not out_path.exists():
            clips.append((rel, out_path))
    if args.max_clips:
        clips = clips[: args.max_clips]
    print(f"{len(clips)} clips to retarget, {args.workers} workers")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUT_ROOT / "qc_report.jsonl"
    n_pass = n_fail = 0
    t0 = time.time()
    with mp.Pool(args.workers, initializer=worker_init,
                 initargs=(not args.subject_shape, args.canonical_height)) as pool, \
            open(report_path, "a") as rep_f:
        for i, rep in enumerate(pool.imap_unordered(process_clip, clips, chunksize=1)):
            rep.pop("trace", None) if rep.get("fail") != "error" else None
            rep_f.write(json.dumps(rep) + "\n")
            rep_f.flush()
            if rep.get("fail") is None:
                n_pass += 1
            else:
                n_fail += 1
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(clips) - i - 1) / rate / 60
                print(f"[{i+1}/{len(clips)}] pass {n_pass} fail {n_fail} "
                      f"(ETA {eta:.0f} min)", flush=True)
    print(f"done: {n_pass} passed, {n_fail} failed -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
