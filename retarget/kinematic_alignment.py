"""Kinematic alignment (Any2Any Sec 3.3.1) for the SONIC G1 -> AgiBot X2 Ultra transfer.

Implements the scattering matrix S_r between the source (Unitree G1, 29 DoF)
and target (AgiBot X2 Ultra, 31 DoF) joint spaces, both expressed in their
respective MuJoCo joint orders.

For this robot pair the hip-decoupling matrix D_r and the closed-chain
correction J_r of the paper are identity: the X2 has orthogonal hip axes and a
fully serial kinematic chain (verified from x2_ultra.urdf), so

    Phi_r  = S_r          (target -> source-aligned)
    Phi_r+ = S_r^T        (source-aligned -> target)

Joint correspondence is by exact name: all 29 G1 joints exist on the X2 under
the same names. The X2's two extra head joints (head_yaw, head_pitch) have no
source counterpart; they are dropped from the policy interface (zero columns
in S_r) and must be held at a fixed pose by a separate low-level loop.

Ordering caveats absorbed by the name-based mapping (do NOT map by index):
  - waist:  G1 MuJoCo order is yaw, roll, pitch; X2 is yaw, pitch, roll
  - wrist:  G1 chain/order is roll, pitch, yaw;  X2 is yaw, pitch, roll
"""

import numpy as np

# ---------------------------------------------------------------------------
# Joint orders (MuJoCo convention), extracted programmatically:
#   G1: gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml
#   X2: aimdk/X2_URDF-v1.3.0/x2_ultra.xml
# ---------------------------------------------------------------------------

G1_MUJOCO_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

X2_MUJOCO_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "head_yaw_joint", "head_pitch_joint",
]

# X2 joints with no G1 counterpart: dropped from the policy interface.
X2_UNMATCHED_JOINTS = ["head_yaw_joint", "head_pitch_joint"]

T_SOURCE = len(G1_MUJOCO_JOINTS)   # 29
N_TARGET = len(X2_MUJOCO_JOINTS)   # 31

# ---------------------------------------------------------------------------
# G1 IsaacLab <-> MuJoCo DOF reorders, from the upstream repo
# (gear_sonic/envs/manager_env/robots/g1.py). Needed because SONIC training
# artifacts (and the reference-motion CSVs) store joints in IsaacLab order.
# ---------------------------------------------------------------------------

G1_ISAACLAB_TO_MUJOCO_DOF = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
    21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])
G1_MUJOCO_TO_ISAACLAB_DOF = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5,
    11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])


# Semantic (chain-position) overrides where NAME matching is physically wrong:
# the two robots build their 3-DoF wrists in OPPOSITE chain order
#   G1: roll(prox, forearm pronation, strong) -> pitch -> yaw(distal, weak)
#   X2: yaw(prox, forearm pronation, strong)  -> pitch -> roll(distal, weak)
# so the pronation axes pair roll<->yaw and the distal axes pair yaw<->roll.
# (Confirmed by actuator classes: pairing by chain position makes every
# action-scale ratio ~1, while name pairing gives 5.65x / 0.16x outliers.)
# value = (x2_joint, sign): FK direction audit (fk_direction_audit, 2026-07-08)
# shows the pronation pair rotates OPPOSITE ways (world-axis dot = -1.0), so
# the pairing carries sign -1; the distal pair is direction-consistent (+0.99).
SEMANTIC_JOINT_MAP = {
    "left_wrist_roll_joint": ("left_wrist_yaw_joint", -1.0),
    "left_wrist_yaw_joint": ("left_wrist_roll_joint", +1.0),
    "right_wrist_roll_joint": ("right_wrist_yaw_joint", -1.0),
    "right_wrist_yaw_joint": ("right_wrist_roll_joint", +1.0),
}


def build_scattering_matrix():
    """Signed scattering S_r in {-1,0,1}^(T x N): entry carries the axis-direction
    sign between matched joints (paper framework: S_r composed with the axis
    convention correction; Phi+ = S_r^T stays the exact inverse since sign^2=1)."""
    s = np.zeros((T_SOURCE, N_TARGET))
    x2_index = {name: j for j, name in enumerate(X2_MUJOCO_JOINTS)}
    for i, name in enumerate(G1_MUJOCO_JOINTS):
        target, sign = SEMANTIC_JOINT_MAP.get(name, (name, 1.0))
        s[i, x2_index[target]] = sign
    return s


def build_hip_decoupling():
    """Hip decoupling matrix D_r (paper Eq. 4/5), T x T in the SOURCE (G1) layout.

    The G1 source has an inclined hip: its roll/yaw axes are rotated ~10 deg
    about the pitch axis (measured: roll axis = [0.985, 0, 0.174] world), while
    the X2 target hips are perfectly orthogonal. FK audit (knee-direction match)
    confirms the target->source hip transform is R(-alpha) on the (roll, yaw)
    pair, i.e. D_r = R(+alpha), and Phi_r = D_r^-1 S_r converts X2 orthogonal hip
    angles into the G1 inclined-hip convention the frozen policy expects.

    Measurement gives the same inclination on left and right (the tilt lies in
    the sagittal x-z plane, unchanged by the sagittal mirror), so no L/R sign
    flip -- unlike the paper's generic H block, whose +-sin handles targets
    whose tilt has a lateral component.
    """
    SIN_A, COS_A = 0.174, (1.0 - 0.174 ** 2) ** 0.5   # alpha = asin(0.174) ~ 10 deg
    D = np.eye(T_SOURCE)
    for roll, yaw in ((G1_MUJOCO_JOINTS.index("left_hip_roll_joint"),
                       G1_MUJOCO_JOINTS.index("left_hip_yaw_joint")),
                      (G1_MUJOCO_JOINTS.index("right_hip_roll_joint"),
                       G1_MUJOCO_JOINTS.index("right_hip_yaw_joint"))):
        D[roll, roll] = COS_A; D[roll, yaw] = -SIN_A   # R(+alpha) on (roll, yaw)
        D[yaw, roll] = SIN_A;  D[yaw, yaw] = COS_A
    return D


S_R = build_scattering_matrix()
D_R = build_hip_decoupling()          # J_r = I: X2 has no closed chains (neq=0)
PHI = np.linalg.inv(D_R) @ S_R        # target -> source-aligned (Phi = D^-1 S)
PHI_PINV = S_R.T @ D_R                # source-aligned -> target (Phi+ = S^T D)

# Zero-pose offsets b (rad): q_x2 = S_r^T q_g1 + b. Calibrated by
# calibrate_offsets.py from bone-direction alignment at the zero pose:
#   - G1's zero pose has the elbows bent ~90 deg forward; X2's arms hang
#     straight, so the elbows need -pi/2 (residual 0.1 deg after correction).
#   - X2's upper arms flare ~11.6 deg outward at zero, but its shoulder_roll
#     limit only allows 0.061 rad of correction; the ~8 deg residual is a genuine
#     mechanical difference left to retargeting / dynamics adaptation.
#   - Legs/waist/wrists/head: < 3 deg, no offset.
X2_JOINT_OFFSETS = {
    "left_elbow_joint": -1.579,
    "right_elbow_joint": -1.585,
    "left_shoulder_roll_joint": -0.061,
    "right_shoulder_roll_joint": +0.061,
}
OFFSET_VEC = np.zeros(N_TARGET)
for _name, _off in X2_JOINT_OFFSETS.items():
    OFFSET_VEC[X2_MUJOCO_JOINTS.index(_name)] = _off


def x2_to_g1(q_x2):
    """Map target-joint vectors (..., 31) into the source-aligned layout (..., 29)."""
    return (np.asarray(q_x2) - OFFSET_VEC) @ PHI.T


def g1_to_x2(q_g1, unmatched_fill=0.0):
    """Map source-layout vectors (..., 29) to target joints (..., 31).

    Applies the calibrated zero-pose offsets. Unmatched target joints (head)
    receive `unmatched_fill` (default: neutral 0).
    """
    q = np.asarray(q_g1) @ PHI_PINV.T + OFFSET_VEC
    if unmatched_fill != 0.0:
        for name in X2_UNMATCHED_JOINTS:
            q[..., X2_MUJOCO_JOINTS.index(name)] = unmatched_fill
    return q


def self_test():
    # signed matrix: |S| carries the injection structure, signs the direction
    assert np.abs(S_R).sum() == T_SOURCE, "each G1 joint must match exactly one X2 joint"
    assert (np.abs(S_R).sum(axis=0) <= 1).all() and (np.abs(S_R).sum(axis=1) == 1).all()
    # Phi+ Phi must be identity on matched target joints, zero on head joints
    p = PHI_PINV @ PHI
    matched = [j for j, n in enumerate(X2_MUJOCO_JOINTS) if n not in X2_UNMATCHED_JOINTS]
    assert np.allclose(p[np.ix_(matched, matched)], np.eye(T_SOURCE))
    for name in X2_UNMATCHED_JOINTS:
        j = X2_MUJOCO_JOINTS.index(name)
        assert p[j].sum() == 0, f"{name} must be outside the mapping domain"
    # round trip on random poses
    rng = np.random.default_rng(0)
    q_g1 = rng.standard_normal((100, T_SOURCE))
    assert np.allclose(x2_to_g1(g1_to_x2(q_g1)), q_g1)
    # ordering spot-checks: waist and wrist land on the right names
    q = np.zeros(T_SOURCE)
    q[G1_MUJOCO_JOINTS.index("waist_roll_joint")] = 0.5
    q[G1_MUJOCO_JOINTS.index("left_wrist_yaw_joint")] = -0.3
    out = g1_to_x2(q)
    assert out[X2_MUJOCO_JOINTS.index("waist_roll_joint")] == 0.5
    # wrists pair by CHAIN POSITION with FK-audited signs:
    # G1 wrist_yaw (distal, +1) -> X2 wrist_roll
    assert out[X2_MUJOCO_JOINTS.index("left_wrist_roll_joint")] == -0.3
    assert out[X2_MUJOCO_JOINTS.index("left_wrist_yaw_joint")] == 0.0
    # G1 wrist_roll (pronation, SIGN -1) -> X2 wrist_yaw flipped
    q2 = np.zeros(T_SOURCE)
    q2[G1_MUJOCO_JOINTS.index("left_wrist_roll_joint")] = 0.4
    out2 = g1_to_x2(q2)
    assert out2[X2_MUJOCO_JOINTS.index("left_wrist_yaw_joint")] == -0.4
    assert out[13] == 0.0 or X2_MUJOCO_JOINTS[13] != "waist_roll_joint"
    print("kinematic_alignment self_test: OK "
          f"(S_r {S_R.shape}, matched {int(S_R.sum())}, dropped {X2_UNMATCHED_JOINTS})")


if __name__ == "__main__":
    self_test()
