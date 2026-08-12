"""Headless SMPL-X (AMASS) -> AgiBot X2 retargeting via GMR.

Wraps GMR's offline pipeline without the interactive viewer: saves the
retargeted motion as a pickle (root pose + joint positions per frame, 30 fps)
and optionally renders an offscreen preview video.

Run inside the `gmr` conda env from the GMR repo root, e.g.:
  python ../Any2Any/retarget/retarget_x2.py --smplx_file <motion_stageii.npz> \
      --out <out.pkl> --video <preview.mp4>
"""

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np

ANY2ANY_ROOT = pathlib.Path(__file__).resolve().parent.parent
GMR_ROOT = ANY2ANY_ROOT.parent / "GMR"
SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", required=True)
    parser.add_argument("--robot", default="agibot_x2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--video", default=None, help="optional offscreen preview mp4")
    parser.add_argument("--subject-shape", action="store_true",
                        help="preserve source SMPL-X betas (legacy)")
    parser.add_argument("--canonical-height", type=float, default=1.8)
    parser.add_argument("--no-ground-snap", action="store_true")
    parser.add_argument("--allow-qc-fail", action="store_true",
                        help="write output even when production QC rejects it")
    args = parser.parse_args()

    if args.robot != "agibot_x2":
        parser.error("the production safety stack currently supports agibot_x2 only")
    input_path = pathlib.Path(args.smplx_file).resolve()
    output_path = pathlib.Path(args.out).resolve()
    video_path = pathlib.Path(args.video).resolve() if args.video else None

    # Reuse exactly the same loader, collision constraints, settle-in, rate
    # limiter, ground model and QC as the batch pipeline.  The previous
    # standalone path silently produced a different class of references.
    try:
        from . import batch_retarget_x2 as batch
    except ImportError:  # direct script execution
        import batch_retarget_x2 as batch

    stdout = sys.stdout
    batch.worker_init(
        canonical=not args.subject_shape,
        canonical_height=args.canonical_height,
    )
    sys.stdout = stdout

    if batch._worker["canonical"]:
        smplx_data, body_model, smplx_output, human_height = batch._load_canonical(
            str(input_path),
        )
    else:
        smplx_data, body_model, smplx_output, human_height = batch._load_subject_segment(
            str(input_path),
        )
    frames, fps = batch._worker["frames"](
        smplx_data, body_model, smplx_output, tgt_fps=30)
    print(f"human height {human_height:.2f} m, {len(frames)} frames @ {fps} fps")

    qpos = batch.retarget_frames(frames, fps, human_height)
    if not args.no_ground_snap:
        qpos, snap_offset = batch.ground_snap(qpos)
        print(f"ground snap: {snap_offset:+.4f} m")
    report = batch.qc_checks(qpos, fps)
    print(f"QC: {report}")
    if report["fail"] is not None and not args.allow_qc_fail:
        raise RuntimeError(
            f"retargeted motion failed QC: {report['fail']} "
            "(use --allow-qc-fail only for diagnosis)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({
            "robot": args.robot,
            "fps": fps,
            "root_pos": qpos[:, :3],
            "root_rot": qpos[:, 3:7],   # wxyz
            "dof_pos": qpos[:, 7:],
            "source": str(input_path),
        }, f)
    print(f"saved {output_path}  (dof dim = {qpos.shape[1] - 7})")

    if video_path:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import cv2
        import mujoco
        from general_motion_retargeting.params import ROBOT_XML_DICT
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        data = mujoco.MjData(model)
        r = mujoco.Renderer(model, 480, 480)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 2.5, -15, 135
        vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (480, 480))
        for t in range(0, len(qpos), 2):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = qpos[t, :3]
            r.update_scene(data, cam)
            vw.write(cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"video written: {video_path}")


if __name__ == "__main__":
    main()
