# X2 retarget: how the current implementation works (handover notes)

Written for anyone continuing in this workspace.  The retarget stack was frozen
and fully validated on 2026-07-23; the snapshot is on the `Retarget` branch of
github.com/TaiE64/ByteGreece.  **Before changing anything described here, run
the "Validation discipline" section, and do not break a red line without the
user's agreement.**

## Architecture (per frame, GMR/general_motion_retargeting/motion_retarget.py)

```
SMPL-X human pose
 -> (1) scaling, scale_human_data (legs 0.68 / arms 0.75, height normalized)
 -> (2) frame offsets, offset_human_data (per-part calibration quaternions,
        smplx_to_x2.json)
 -> (3) arms = _x2_solve_arms_analytic (final version, 2026-07-23):
        The shoulder tracks the visible SMPL shoulder->elbow bone direction; the
        weak forearm direction only removes the twist null space.  The elbow
        angle is taken directly from the shoulder/elbow/wrist angle, and only the
        wrist pose uses the joint rotation relative to spine3.  The shoulder
        triangle is solved deterministically from qpos0 every frame, so
        warm-start cannot accumulate into an equivalent but contorted Euler
        branch; a weak qpos0 tie-break picks the canonical branch inside the
        axial twist null space, where the geometry error is equal either way.
        It is not a named-motion prior.  The unobservable axial twist of the raw
        shoulder joint frame no longer reaches X2 (the reference hit +/-150 deg
        during the raised-arm section of take_01).  The output still has the
        35 deg/frame solver branch guard and the shared 12 rad/s safety clamp,
        and no low-pass.  The QP arm task weights are zeroed; when the source
        lacks SMPL geometry or rotation it falls back to the position path,
        which also had its blend-based temporal filter removed.
 -> (3b) collision projection, _x2_project_collisions: the analytic arm bypasses
        the QP collision limits, so afterwards the whole body is pinned and a
        temporary negative bound_relaxation iteratively pushes penetration out
        (mink's collision limit only prevents penetration, it cannot undo it).
        walk selfcol 0.16 -> 0.04
 -> (4) foot bias removal, _x2_foot_orientation_targets: a per-subject constant
        sole-tilt bias (batch: precalibrate_x2_foot_bias over the whole clip,
        then frozen; streaming: a slow online estimate, with vertical velocity
        deciding when the foot is still)
 -> (5) two-stage weighted IK (mink QP; weights in the json; pelvis rot=100
        anchors the root)
 -> (6) arm branch recovery, _x2_recover_stuck_arms, on two triggers:
        a) residual >10 cm: the reconstructed target is reachable by
           construction, so a large steady-state residual means the QP is stuck
           in a wrong branch from warm-start.  Fast motion drags the shoulder
           chain past a singularity and it never recovers on its own (130+
           frames of arms frozen behind and out after the boxing section of
           take1).
        b) an arm-chain joint sits on a hard limit (<0.02 rad) with residual
           >5 cm: the "wound-up" branch can get the position under the threshold
           of (a) and hide there (the right shoulder pitch pinned at its +116.9
           deg limit during the walk section of take1, residual 7.6 cm against a
           healthy 2.6, forcing the right elbow 37 deg more bent than the
           reference -- the "zombie right arm" the user saw; the left arm in the
           same pose sat at pitch 26 deg / roll -3.5 deg).
        Recovery resets that arm chain to qpos0, re-solves the same frame and
        keeps whichever residual is smaller, restoring the original on failure.
        This is a pure solver-level mechanism: not a filter and not a prior, and
        it never fires on healthy frames (QC metrics over 5 clips are bit
        identical with it disabled).  Afterwards both elbows correlate 0.94/0.94
        on take1, and the batch chain has zero frames with forearm direction
        error >40 deg.  The live chain and the batch chain share the same
        12 rad/s joint velocity safety clamp, with no low-pass.
```

Batch entry point, Any2Any/retarget/batch_retarget_x2.py:
- calls `rt.precalibrate_x2_foot_bias(frames)` and then retargets frame by frame
- output has **no low-pass filtering**, only the 12 rad/s single-frame safety
  clamp, which fires only on IK glitches

## Red lines (explicitly required by the user; violating one means a rollback)

1. **No low-pass or constant-gain filter on the reference output.**  A 0.35
   filter once cut the punch elbow speed from 22.9 to 8.0 deg/frame, eating two
   thirds of the fast motion.  Offline and live keep only the shared 12 rad/s
   joint velocity safety clamp.
2. **No named-motion pose priors.**  The shoulder qpos0 term only picks the
   canonical Euler branch inside the axial twist null space that the bone
   direction leaves unconstrained, at a twentieth of the main direction weight.
   It must not be used for motion smoothing or to pull toward a hand-made pose.
3. **No hardcoded correction angles.**  The foot bias must be solved from the
   data per subject; individual arch and toe-out differences span +/-5-20 deg,
   so any constant is wrong.

## Validation discipline (everything must be green before you finish)

```bash
# unit tests (gmr env)
cd Mimic && python -m unittest tests.test_x2_retarget_tpose
cd Any2Any && python -c "<run retarget/tests/test_retarget_regressions.py>"
# batch QC (decisive; unit tests do not catch whole-clip problems)
python retarget/batch_retarget_x2.py --clip_list <5 clips> --out_root /tmp/audit
# the 5 clips: punchboxing_kick / jumping_jacks / squat_simple / normal_walk1 / shake_arms
```

Quantitative baselines, aligned against the human source (a regression is any
degradation; the bone-direction arm solver applies from 2026-07-23):
- arm evaluation = eval_arm_tracking (residual after a constant Wahba rotation
  fit): take1 upper-arm residual median L/R <=4/3 deg, elbow angle RMSE <=2 deg,
  elbow correlation >=0.99; the shoulder pitch/yaw no longer strands at a
  +/-150 deg branch after a motion ends
- punch elbow speed 22.9 deg/frame (the human does 25.2); walk knee speed 12.2,
  which must stay <=14 with no jump over 15 deg
- jumping-jacks waist roll std ~1.4 deg (the old 7.1 was an arm-borrowing
  artifact; a symmetric motion should not rock the waist)
- squat support-foot sole tilt <2 deg; walk selfcol <=0.04 (a light graze
  against a 0.10 threshold)

## Open items (ask before touching)

1. Treadmill-style data (BMLrub) may not sample the single-foot bias well enough
   (walk leaves ~12 deg on the right foot)
2. bvh_lafan1_to_x2.json does not match the smplx config (elbows unexpectedly hit
   their limits on ~5% of frames); it must be fixed before regenerating LAFAN
   training data
3. QC penalises "naturally straight arm held for a while" too harshly, which
   kills conversational gesture clips

## Lessons (do not repeat)

- The 180 deg structure of the right-arm calibration quaternion relative to the
  left is **correct**; do not "fix" it by mirroring
- Proportional scaling cannot fix the arm proportion difference -> bone lengths
  must be rebuilt.  Legs do not need it (knee and hip position weights are 0,
  only the foot is an anchor, and the knee angle absorbs the length difference)
- Height-gated foot flattening switches targets during the gait phase and
  excites the knee (measured 22.9 deg/frame jumps) -> remove a constant bias
  instead of flattening as a function of height
- "The single-frame metric looks fine" is an illusion: take1's stuck-branch arm
  section solves correctly when each frame is re-solved cold in isolation, and
  only reproduces when the sequence is replayed.  Diagnosing a stuck-solver
  problem requires comparing the sequential solve against the isolated one
- The automatic calibration in the Roboparty GMR branch was evaluated and
  rejected (the scaling optimizer degenerates)
