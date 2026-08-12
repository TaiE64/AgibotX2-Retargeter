# GMR X2 overlay

The retargeter runs on a locally patched [YanjieZe/GMR](https://github.com/YanjieZe/GMR)
fork. This directory carries everything we authored on top of upstream:

- `UPSTREAM` — the upstream commit our patch applies to
- `gmr_upstream.patch` — our fixes to upstream GMR (`git apply gmr_upstream.patch`
  from the GMR repo root). Notable: `solve_ik` previously passed `limits` as
  `safety_break`, so IK limits were never active; SMPL-X Y-up -> Z-up
  normalization in `utils/smpl.py`.
- `ik_configs/` — the calibrated X2 retargeting tables (copy into
  `general_motion_retargeting/ik_configs/`):
  - `smplx_to_x2.json` — SMPL-X (AMASS) -> X2, the production config.
    The 180-deg right-arm rotation offsets are CORRECT (X2's left/right joint
    ranges are mirrored); do not "fix" them by symmetry arguments.
  - `bvh_lafan1_to_x2.json` — LAFAN1 BVH -> X2 (evaluation sets)
  - `bvh_xsens_to_x2.json` — Xsens live streaming -> X2

Not included (not ours to redistribute):

- `assets/agibot_x2/` (X2 MJCF + STL meshes) — converted from the AgiBot X2
  vendor URDF; obtain the URDF from AgiBot and convert, or copy from an
  internal checkout.
- SMPL-X body models (`assets/body_models/`) — download from smpl-x.is.tue.mpg.de
  under their license.
