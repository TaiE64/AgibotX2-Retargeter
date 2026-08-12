"""Generate bvh_lafan1_to_x2.json for GMR (Any2Any evaluation set).

Strategy: start from bvh_lafan1_to_g1.json (validated LAFAN source-side
skeleton handling) but replace the ARM entries with the position-dominant
tracking scheme of smplx_to_x2.json (validated on 9,380 AMASS->X2 clips).
Rotation-dominant arm tracking (wrist pos weight 0 / rot 10) causes IK
solution-branch flips on fast LAFAN motion (single-frame ~1.7 rad
shoulder-yaw jumps); position-dominant tracking has no branches to flip.

Frame bookkeeping: the rot-offset columns map a human body frame to a robot
link frame. For the same anatomical bone tracked in both the LAFAN (L) and
SMPLX (S) conventions against the SAME G1 link, o_L = q_LS o_S, so
q_LS = o_L o_S^-1 is the bone's L->S frame change, independent of robot.
The X2 entry then transplants as  o'_L = q_LS o'_S,  off'_L = R(q_LS) off'_S.
(X2 link frames follow G1 conventions: all 15 shared smplx offsets are
identical between smplx_to_g1 and smplx_to_x2.)

Run (gmr env, any cwd):  python gen_lafan_x2_ik_config.py
"""

import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R

IK_DIR = (pathlib.Path(__file__).resolve().parent.parent / "GMR" /
          "general_motion_retargeting" / "ik_configs")

# lafan bone <-> smplx bone (arm chain + hip, the transplanted entries)
BONE_MAP = {
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}
# G1 links shared by bvh_lafan1_to_g1 and smplx_to_g1 -> per-bone q_LS bridge
BRIDGE_LINK = {
    "LeftArm": "left_shoulder_yaw_link", "RightArm": "right_shoulder_yaw_link",
    "LeftForeArm": "left_elbow_link", "RightForeArm": "right_elbow_link",
    "LeftHand": "left_wrist_yaw_link", "RightHand": "right_wrist_yaw_link",
}
# X2 arm links to transplant from smplx_to_x2 (hand-end differs by chain pos)
X2_ARM = {
    "left_shoulder_yaw_link": "left_shoulder_yaw_link",
    "right_shoulder_yaw_link": "right_shoulder_yaw_link",
    "left_elbow_link": "left_elbow_link",
    "right_elbow_link": "right_elbow_link",
    "left_wrist_roll_link": "left_wrist_roll_link",
    "right_wrist_roll_link": "right_wrist_roll_link",
}

ARM_BONES = {"LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
             "LeftHand", "RightHand"}


def q_mul(a, b):
    """Hamilton product, [w,x,y,z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rot(q, v):
    return R.from_quat(np.roll(q, -1)).apply(v)  # wxyz -> xyzw


def main():
    lg = json.load(open(IK_DIR / "bvh_lafan1_to_g1.json"))
    sg = json.load(open(IK_DIR / "smplx_to_g1.json"))
    sx = json.load(open(IK_DIR / "smplx_to_x2.json"))

    out = json.loads(json.dumps(lg))  # deep copy of the lafan->g1 base

    for tbl in ("ik_match_table1", "ik_match_table2"):
        t = out[tbl]
        # base swap: X2's hand-end link is wrist_roll (chain position)
        for side in ("left", "right"):
            t[f"{side}_wrist_roll_link"] = t.pop(f"{side}_wrist_yaw_link")
        # transplant arm entries from smplx_to_x2 via the q_LS bone bridge
        for x2_link in X2_ARM:
            sx_entry = sx[tbl][x2_link]
            smplx_bone, wp, wr, off_s, quat_s = sx_entry
            lafan_bone = BONE_MAP[smplx_bone]
            bridge = BRIDGE_LINK[lafan_bone]
            o_l = np.array(lg[tbl][bridge][4])
            o_s = np.array(sg[tbl][bridge][4])
            q_ls = q_mul(o_l, q_conj(o_s))
            quat_l = q_mul(q_ls, np.array(quat_s))
            off_l = rot(q_ls, np.array(off_s, dtype=float))
            t[x2_link] = [lafan_bone, wp, wr,
                          [round(float(v), 4) for v in off_l],
                          [round(float(v), 8) for v in quat_l]]

    # size scaling: multiply lafan->g1 scales by the X2/G1 per-limb ratio
    ARM_HUMAN = {"LeftArm", "LeftForeArm", "LeftHand",
                 "RightArm", "RightForeArm", "RightHand"}
    for k, v in lg["human_scale_table"].items():
        out["human_scale_table"][k] = round(
            v * (0.75 / 0.8 if k in ARM_HUMAN else 0.68 / 0.9), 4)

    path = IK_DIR / "bvh_lafan1_to_x2.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"written {path}")
    for tbl in ("ik_match_table1",):
        for k in sorted(X2_ARM):
            print(f"  {k}: {out[tbl][k]}")


if __name__ == "__main__":
    main()
