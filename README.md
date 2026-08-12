# AgibotX2 Retargeter

Human motion → **AgiBot X2 Ultra** (31 DoF) whole-body retargeting.
Takes SMPL-X (AMASS), LAFAN1 (BVH) or Xsens streams and produces
physically-screened X2 joint trajectories (root pose + 31 DoF @ 30 Hz),
ready for tracking-policy training or kinematic playback.

Battle-tested as the data engine of the SONIC→X2 transfer
([Sonic2X2](https://github.com/TaiE64/Sonic2X2)); split out here as a
standalone, reusable component.

## How it works

```
SMPL-X / BVH ──► GMR (mink IK, calibrated X2 tables) ──► rate limiter ──► ground snap ──► QC ──► pkl
```

- **NativeGMR** (`retarget/native_gmr.py`): GMR/mink frame tasks driven by the
  calibrated `smplx_to_x2` table — pelvis + feet position anchors (w=100),
  elbow/wrist position + orientation tasks, plus a posture task whose per-joint
  costs pin the X2's Euler-shoulder and waist-pitch null spaces to the sane
  IK branch (the ±π shoulder-roll branch is reachable from ordinary poses
  otherwise).
- **Collision limits inside the IK QP**: arm-vs-lower-body and hand-vs-hand
  minimum-distance constraints (requires the `solve_ik(limits=...)` upstream
  fix carried in `gmr_x2_overlay/gmr_upstream.patch` — stock GMR silently
  dropped IK limits).
- **Trajectory rate limiter** (12 rad/s): converts single-frame IK branch
  flips into bounded sweeps and feeds the clamped pose back so the solver
  stays on the incumbent branch. No low-pass filtering — fast transients
  (punches) keep their velocity.
- **Ground snap**: FK sole heights, 5th percentile pinned to z=0.
- **Per-clip QC** (`retarget/retarget_quality.py` + batch pipelines):
  finite check, joint/root velocity ceilings, sole penetration/float,
  self-collision fraction, per-joint hard-limit dwell with an
  expected-at-limit whitelist, root jump rejection.

## Entry points

```bash
# single clip + optional preview video (gmr env, from the GMR repo root)
python retarget/retarget_x2.py --smplx_file <motion_stageii.npz> --out out.pkl

# batch AMASS with QC manifest
python retarget/batch_retarget_x2.py --subsets ACCAD CMU --workers 8 --out_root <dir>

# LAFAN / OMOMO / 100STYLE evaluation sets
python retarget/batch_retarget_lafan_x2.py ...
python retarget/batch_retarget_omomo_x2.py ...

# command-labelled locomotion references (contact-consistent root XY)
python retarget/build_locomotion_reference.py --src <retargeted> --out-root <dir>
```

Output schema: `{robot, fps, root_pos (T,3), root_rot (T,4 wxyz), dof_pos (T,31), source}`.

## Calibration tables (`gmr_x2_overlay/`)

Our deltas on top of [YanjieZe/GMR](https://github.com/YanjieZe/GMR) — apply
`gmr_upstream.patch` at the commit recorded in `UPSTREAM`, drop `ik_configs/`
in place. The right-arm 180° rotation offsets in the tables are **correct**
(X2's left/right joint ranges are mirrored); see `docs/RETARGET_NOTES.md`
before touching anything.

## Validated behaviour (2026-08)

- Walking/running/martial-arts/dance clips track the human reference
  frame-by-frame (elbow correlation ≥0.99 on benchmarks; see notes).
- 214-clip AMASS QC sample: 59% pass; rejections are dominated by genuine
  hardware-envelope violations (the X2 waist pitch is ±18° — deep bends,
  squats and crawls cannot be represented and are correctly rejected).

## What you must bring (not redistributable)

- AgiBot X2 URDF/meshes (vendor) → the MJCF under GMR `assets/agibot_x2/`
- SMPL-X body models — smpl-x.is.tue.mpg.de
- AMASS / LAFAN1 / OMOMO motion data (registration-gated, research terms)

## License

Apache-2.0 (this repository's own code). Upstream GMR and all data/assets
keep their own licenses.
