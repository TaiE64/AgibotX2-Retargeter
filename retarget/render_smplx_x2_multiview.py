"""Render frame-aligned SMPL-X and X2 front/side validation video."""

import argparse
import os
import pathlib
import pickle
import sys

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402


HERE = pathlib.Path(__file__).resolve().parent
SR_ROOT = HERE.parent.parent
GMR_ROOT = SR_ROOT / "GMR"
RETARGET_ROOT = SR_ROOT / "Any2Any" / "retarget"
sys.path.insert(0, str(GMR_ROOT))
sys.path.insert(0, str(RETARGET_ROOT))

import batch_retarget_x2 as batch  # noqa: E402
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402


JOINTS = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist",
)


def draw_reference(frame, parents, ground_z, side_view, width, height):
    points = np.stack([frame[name][0] for name in JOINTS])
    center = points[0]
    scale = 220.0
    horizontal_axis = 1 if side_view else 0
    horizontal = points[:, horizontal_axis] - center[horizontal_axis]
    if side_view:
        horizontal = -horizontal
    pixels = np.column_stack((
        width * 0.5 + scale * horizontal,
        height - 38 - scale * (points[:, 2] - ground_z),
    )).astype(np.int32)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    ground_y = int(height - 38)
    cv2.line(image, (18, ground_y), (width - 18, ground_y),
             (70, 70, 70), 1, cv2.LINE_AA)
    for child, parent in enumerate(parents):
        if parent >= 0:
            cv2.line(image, tuple(pixels[parent]), tuple(pixels[child]),
                     (30, 80, 255), 4, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(image, tuple(point), 5, (255, 80, 20), -1, cv2.LINE_AA)
    return image


def label(image, text, frame_index):
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width - 1, 42), (12, 12, 12), -1)
    cv2.putText(image, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (0, 170, 255), 2, cv2.LINE_AA)
    frame_text = f"FRAME {frame_index:04d}"
    text_width = cv2.getTextSize(
        frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    )[0][0]
    cv2.rectangle(
        image, (width - text_width - 22, height - 27),
        (width - 1, height - 1), (12, 12, 12), -1,
    )
    cv2.putText(
        image, frame_text, (width - text_width - 12, height - 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (0, 0), (width - 1, height - 1),
                  (65, 65, 65), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-file", required=True)
    parser.add_argument("--robot-motion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--layout", choices=("row", "grid"), default="row",
        help="Use a four-pane row or a front/side 2x2 comparison grid.",
    )
    parser.add_argument(
        "--allow-trim", action="store_true",
        help="Trim a one-frame loader mismatch for cross-fork comparisons.",
    )
    args = parser.parse_args()

    source_path = pathlib.Path(args.smplx_file).resolve()
    motion_path = pathlib.Path(args.robot_motion).resolve()
    output_path = pathlib.Path(args.output).resolve()

    original_stdout = sys.stdout
    batch.worker_init(canonical=True, canonical_height=1.8)
    sys.stdout = original_stdout
    smplx_data, body_model, smplx_output, _ = batch._load_canonical(
        str(source_path)
    )
    frames, source_fps = batch._worker["frames"](
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    parents = np.asarray(body_model.parents[:len(JOINTS)], dtype=np.int32)

    with motion_path.open("rb") as motion_file:
        motion = pickle.load(motion_file)
    fps = float(motion["fps"])
    qpos = np.column_stack((
        motion["root_pos"], motion["root_rot"], motion["dof_pos"],
    ))
    if len(frames) != len(qpos) and args.allow_trim:
        frame_count = min(len(frames), len(qpos))
        frames = frames[:frame_count]
        qpos = qpos[:frame_count]
    if len(frames) != len(qpos) or abs(source_fps - fps) > 1e-6:
        raise ValueError(
            f"source/robot mismatch: {len(frames)}@{source_fps:g} vs "
            f"{len(qpos)}@{fps:g}"
        )

    feet_z = np.asarray([
        frame[name][0][2]
        for frame in frames
        for name in ("left_ankle", "right_ankle", "left_foot", "right_foot")
    ])
    ground_z = float(np.percentile(feet_z, 5))

    pane_width, height = ((480, 480) if args.layout == "grid"
                          else (360, 480))
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["agibot_x2"]))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, pane_width)
    front_camera = mujoco.MjvCamera()
    front_camera.distance, front_camera.elevation, front_camera.azimuth = (
        2.55, -5, 90
    )
    side_camera = mujoco.MjvCamera()
    side_camera.distance, side_camera.elevation, side_camera.azimuth = (
        2.55, -5, 0
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_size = ((pane_width * 2, height * 2)
                   if args.layout == "grid"
                   else (pane_width * 4, height))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {output_path}")

    title = args.title or source_path.stem.removesuffix("_stageii")
    for frame_index, (frame, configuration) in enumerate(zip(frames, qpos)):
        ref_front = draw_reference(
            frame, parents, ground_z, False, pane_width, height
        )
        ref_side = draw_reference(
            frame, parents, ground_z, True, pane_width, height
        )

        data.qpos[:] = configuration
        mujoco.mj_forward(model, data)
        for camera in (front_camera, side_camera):
            camera.lookat[:] = configuration[:3] + np.array([0.0, 0.0, 0.05])
        renderer.update_scene(data, front_camera)
        robot_front = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        renderer.update_scene(data, side_camera)
        robot_side = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        label(ref_front, f"{title} | SMPL REFERENCE | FRONT", frame_index)
        label(ref_side, "SMPL REFERENCE | SIDE", frame_index)
        label(robot_front, "X2 RETARGET REFERENCE | FRONT", frame_index)
        label(robot_side, "X2 RETARGET REFERENCE | SIDE", frame_index)
        if args.layout == "grid":
            video_frame = np.concatenate((
                np.concatenate((ref_front, robot_front), axis=1),
                np.concatenate((ref_side, robot_side), axis=1),
            ), axis=0)
        else:
            video_frame = np.concatenate(
                (ref_front, ref_side, robot_front, robot_side), axis=1
            )
        writer.write(video_frame)

    writer.release()
    renderer.close()
    print(f"saved {len(qpos)} frames @ {fps:g} fps: {output_path}")


if __name__ == "__main__":
    main()
