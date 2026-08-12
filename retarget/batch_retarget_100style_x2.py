"""Batch-retarget the official trimmed 100STYLE clips to AgiBot X2 NPZ50.

The 100STYLE Frame_Cuts stop indices are inclusive.  Each output is a complete
motion_tracking NPZ (xyzw root quaternion and root-local X2 FK positions).

Run from the GMR conda environment, for example:
  python retarget/batch_retarget_100style_x2.py --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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
DEFAULT_DATA_ROOT = pathlib.Path("/data/motion/heft_expansion")
X2_XML = GMR_ROOT / "assets" / "agibot_x2" / "x2_mocap.xml"

SOURCE_FPS = 60
TARGET_FPS = 50
VMAX = 12.0
VEL_LIMIT = 15.0
PENETRATION_LIMIT = -0.06
FLOAT_LIMIT = 0.06
SELFCOL_LIMIT = 0.10

_worker = {}


def _body_geoms(model, mujoco, names):
    geoms = []
    for name in names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geoms.extend(
            geom for geom in range(model.ngeom)
            if model.geom_bodyid[geom] == body and model.geom_contype[geom] != 0
        )
    return geoms


def worker_init():
    sys.path.insert(0, str(GMR_ROOT))
    os.chdir(GMR_ROOT)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.kinematics_model import KinematicsModel
    from general_motion_retargeting.utils.xsens import load_xsens_file
    from mink.limits import CollisionAvoidanceLimit

    model = mujoco.MjModel.from_xml_path(str(X2_XML))
    data = mujoco.MjData(model)
    kinematics = KinematicsModel(str(X2_XML), device="cpu")
    joint_names = [
        model.joint(joint_id).name
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if len(joint_names) != kinematics.num_dof:
        raise RuntimeError("X2 joint-name schema does not match FK DoFs")

    def ancestors(body):
        chain = []
        while body > 0:
            chain.append(body)
            body = model.body_parentid[body]
        chain.append(0)
        return chain

    chains = {body: ancestors(body) for body in range(model.nbody)}
    contact_baseline = set()
    for body1 in range(model.nbody):
        for body2 in range(body1 + 1, model.nbody):
            common = next(item for item in chains[body1] if item in chains[body2])
            if chains[body1].index(common) + chains[body2].index(common) <= 2:
                contact_baseline.add((body1, body2))

    _worker.update(
        torch=torch,
        mujoco=mujoco,
        GMR=GMR,
        load=load_xsens_file,
        CollisionAvoidanceLimit=CollisionAvoidanceLimit,
        model=model,
        data=data,
        kinematics=kinematics,
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(kinematics.body_names),
        contact_baseline=contact_baseline,
        foot_geoms=foot_collision_geoms(model, mujoco),
    )


def _add_collision_avoidance(retargeter):
    model = retargeter.model
    mujoco = _worker["mujoco"]
    arm_l = [f"left_{part}_link" for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]
    arm_r = [f"right_{part}_link" for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")]
    lower = ["pelvis", "torso_link"] + [
        f"{side}_{part}_link"
        for side in ("left", "right")
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
    ]
    geoms_l = _body_geoms(model, mujoco, arm_l)
    geoms_r = _body_geoms(model, mujoco, arm_r)
    geoms_lower = _body_geoms(model, mujoco, lower)
    retargeter.ik_limits.append(
        _worker["CollisionAvoidanceLimit"](
            model,
            [(geoms_l + geoms_r, geoms_lower), (geoms_l, geoms_r)],
            minimum_distance_from_collisions=X2_ARM_COLLISION_MARGIN,
        )
    )


def _resample_qpos(qpos, source_fps=SOURCE_FPS, target_fps=TARGET_FPS):
    duration = (len(qpos) - 1) / float(source_fps)
    count = int(round(duration * target_fps)) + 1
    src_t = np.arange(len(qpos), dtype=np.float64) / float(source_fps)
    dst_t = np.minimum(np.arange(count, dtype=np.float64) / float(target_fps), src_t[-1])
    root_pos = np.stack([np.interp(dst_t, src_t, qpos[:, i]) for i in range(3)], axis=-1)
    root_xyzw = np.concatenate([qpos[:, 4:7], qpos[:, 3:4]], axis=-1)
    root_rot = Slerp(src_t, Rotation.from_quat(root_xyzw))(dst_t).as_quat()
    dof_pos = np.stack([np.interp(dst_t, src_t, qpos[:, i]) for i in range(7, qpos.shape[1])], axis=-1)
    qpos_wxyz = np.concatenate([root_pos, root_rot[:, 3:4], root_rot[:, :3], dof_pos], axis=-1)
    return qpos_wxyz.astype(np.float32), root_rot.astype(np.float32)


def _local_fk(dof_pos):
    torch = _worker["torch"]
    kinematics = _worker["kinematics"]
    root_pos = torch.zeros((len(dof_pos), 3), dtype=torch.float32)
    root_rot = torch.zeros((len(dof_pos), 4), dtype=torch.float32)
    root_rot[:, 3] = 1.0
    with torch.inference_mode():
        body_pos, _ = kinematics.forward_kinematics(
            root_pos, root_rot, torch.from_numpy(dof_pos).float()
        )
    return body_pos.numpy().astype(np.float32)


def ground_snap(qpos):
    mujoco = _worker["mujoco"]
    model, data = _worker["model"], _worker["data"]
    foot_geoms = _worker["foot_geoms"]
    soles = []
    for frame in range(0, len(qpos), 5):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        soles.append(sole_height(model, data, foot_geoms))
    offset = float(np.percentile(np.asarray(soles), 5))
    if abs(offset) > 0.005:
        qpos = qpos.copy()
        qpos[:, 2] -= offset
    return qpos, offset


def qc_checks(qpos, fps):
    mujoco = _worker["mujoco"]
    model, data = _worker["model"], _worker["data"]
    report = {"finite": bool(np.isfinite(qpos).all())}
    if not report["finite"]:
        report["fail"] = "nan"
        return report
    velocity = float(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * fps)
    report["max_root_vel"], report["max_root_ang_vel"] = root_motion_metrics(qpos, fps)
    report["at_limit"], unexpected_limits = joint_limit_report(
        model, mujoco, qpos[:, 7:], fps=fps,
    )
    report["unexpected_at_limit"] = unexpected_limits
    foot_geoms = _worker["foot_geoms"]
    soles, self_collision = [], 0
    sample = list(range(0, len(qpos), 5))
    for frame in sample:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        soles.append(sole_height(model, data, foot_geoms))
        for contact in data.contact[: data.ncon]:
            if contact.dist < -0.015:
                body1, body2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
                pair = (min(body1, body2), max(body1, body2))
                if pair not in _worker["contact_baseline"]:
                    self_collision += 1
                    break
    soles = np.asarray(soles)
    report.update(
        max_joint_vel=velocity,
        sole_min=float(soles.min()),
        sole_stance_med=float(np.percentile(soles, 20)),
        selfcol_frac=round(self_collision / len(sample), 3),
        root_med=float(np.median(qpos[:, 2])),
    )
    grounded_gait = report["root_med"] > 0.45
    if velocity > VEL_LIMIT:
        report["fail"] = "velocity"
    elif report["max_root_vel"] > ROOT_LINEAR_VELOCITY_LIMIT:
        report["fail"] = "root_velocity"
    elif report["max_root_ang_vel"] > ROOT_ANGULAR_VELOCITY_LIMIT:
        report["fail"] = "root_angular_velocity"
    elif unexpected_limits:
        report["fail"] = "joint_limit"
    elif report["sole_min"] < PENETRATION_LIMIT:
        report["fail"] = "penetration"
    elif grounded_gait and report["sole_stance_med"] > FLOAT_LIMIT:
        report["fail"] = "float"
    elif report["selfcol_frac"] > SELFCOL_LIMIT:
        report["fail"] = "self_collision"
    else:
        report["fail"] = None
    return report


def process_clip(item):
    source, style, movement, start, stop, target = item
    started = time.time()
    try:
        loader_args = SimpleNamespace(
            bvh_file=str(source), scale=0.01, start=start, end=stop + 1,
            reset_to_zero=False, bvh_format="3DSM",
        )
        frames, human_height, frame_time = _worker["load"](loader_args)
        source_fps = int(round(1.0 / frame_time))
        if source_fps != SOURCE_FPS:
            raise ValueError(f"expected {SOURCE_FPS} fps, got {source_fps}")
        retargeter = _worker["GMR"](
            actual_human_height=human_height,
            src_human="bvh_xsens",
            tgt_robot="agibot_x2",
            verbose=False,
            use_velocity_limit=True,
        )
        _add_collision_avoidance(retargeter)
        for _ in range(20):
            retargeter.retarget(frames[0])
        step = VMAX / SOURCE_FPS
        qpos, previous = [], None
        for frame in frames:
            # Preserve source flight phases.  Per-frame offset_to_ground pins
            # the lowest human foot independently on every frame and removes
            # the shared ballistic vertical motion; one rigid robot-space
            # ground_snap below is the correct clip-level operation.
            current = retargeter.retarget(frame).copy()
            if previous is not None and np.abs(current[7:] - previous[7:]).max() > step:
                current[7:] = previous[7:] + np.clip(current[7:] - previous[7:], -step, step)
                retargeter.configuration.update(current)
            qpos.append(current)
            previous = current
        qpos = np.stack(qpos)
        qpos, ground_offset = ground_snap(qpos)
        qpos, root_rot_xyzw = _resample_qpos(qpos)
        report = qc_checks(qpos, TARGET_FPS)
        report.update(
            clip=f"{style}_{movement}", source=str(source), source_start=start,
            source_stop=stop, source_frames=stop - start + 1, frames=len(qpos),
            seconds=round(len(qpos) / TARGET_FPS, 3),
            ik_seconds=round(time.time() - started, 2),
            ground_snap_m=round(ground_offset, 4),
        )
        if report["fail"] is None:
            dof_pos = qpos[:, 7:].astype(np.float32)
            local_body_pos = _local_fk(dof_pos)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                root_pos=qpos[:, :3].astype(np.float32),
                root_rot=root_rot_xyzw,
                dof_pos=dof_pos,
                local_body_pos=local_body_pos,
                joint_names=_worker["joint_names"],
                body_names=_worker["body_names"],
                fps=np.int64(TARGET_FPS),
            )
        return report
    except Exception as exc:
        return {
            "clip": f"{style}_{movement}", "source": str(source), "fail": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-800:],
        }


def _load_jobs(source_root, cuts_path, output_root, overwrite=False):
    # The official ZIP contains macOS AppleDouble files under __MACOSX whose
    # names also end in .bvh.  They are 212-byte metadata, not motion files.
    files = {
        path.stem: path
        for path in source_root.rglob("*.bvh")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    }
    jobs, missing = [], []
    with cuts_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            style = row["STYLE_NAME"]
            for movement in ("BR", "BW", "FR", "FW", "ID", "SR", "SW", "TR1", "TR2", "TR3"):
                start, stop = row[f"{movement}_START"], row[f"{movement}_STOP"]
                if start == "N/A" or stop == "N/A":
                    continue
                stem = f"{style}_{movement}"
                if stem not in files:
                    missing.append(stem)
                    continue
                target = output_root / style / f"{movement}.npz"
                if overwrite or not target.exists():
                    jobs.append((files[stem], style, movement, int(start), int(stop), target))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} official BVH files, first entries: {missing[:10]}")
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT / "raw/100style/extracted")
    parser.add_argument("--cuts", type=pathlib.Path, default=DEFAULT_DATA_ROOT / "raw/100style/Frame_Cuts.csv")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT / "retargeted_npz50/100style")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    jobs = _load_jobs(args.source_root, args.cuts, args.output_root, args.overwrite)
    if args.max_clips is not None:
        jobs = jobs[: args.max_clips]
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "qc_report.jsonl"
    passed = failed = 0
    started = time.time()
    print(f"{len(jobs)} clips, {args.workers} workers -> {args.output_root}")
    with mp.Pool(args.workers, initializer=worker_init) as pool, report_path.open("a", encoding="utf-8") as report_file:
        for index, report in enumerate(pool.imap_unordered(process_clip, jobs, chunksize=1), start=1):
            report_file.write(json.dumps(report) + "\n")
            report_file.flush()
            if report.get("fail") is None:
                passed += 1
            else:
                failed += 1
            print(
                f"[{index}/{len(jobs)}] {report['clip']} -> {report.get('fail') or 'PASS'} "
                f"({report.get('ik_seconds', '?')}s)", flush=True,
            )
    print(f"done: passed={passed}, failed={failed}, wall={time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
