"""Unitree G1 standing balance with LQR, in MuJoCo (with GUI).

Recipe follows the MuJoCo LQR tutorial (humanoid balance), adapted to the G1:
  1. Find a standing pose where the feet carry the weight (inverse dynamics
     height search), and the actuator controls that hold it (ctrl0).
  2. Linearize the dynamics around (qpos0, ctrl0) with finite differences.
  3. Cost: keep the centre of mass over the middle of the two feet, plus a
     small cost on joint deviations. Solve the discrete Riccati equation.
  4. Control law: ctrl = ctrl0 - K @ dx.

The G1 menagerie model uses position actuators (kp=500), so ctrl is a joint
target angle. "PD hold" mode (ctrl = ctrl0, no feedback from K) is included
for comparison: it is what the robot does with only its joint PD loops.

Usage (from the repo root):
  python experiments/g1_balance/g1_lqr_balance.py
  ... --push 60            # automatic push magnitude [N] (0 disables)
  ... --test               # headless push test, LQR vs PD hold, no GUI
  ... --save [file.npz]    # save A, B, Q, R, P, K + operating point
                           # (default g1_lqr_matrices.npz next to this script)

GUI keys:
  Insert  push the robot now (random horizontal direction)
  End     toggle controller LQR <-> PD hold
  Home    reset to the standing pose
  Ctrl + right-drag on a body applies your own force (MuJoCo built-in).
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import scipy.linalg
import scipy.optimize

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "external", "unitree_g1", "scene.xml")

# Cost weights (tutorial values, R scaled because ctrl is in radians).
BALANCE_COST = 1000.0  # CoM over the support centre
LEG_JOINT_COST = 3.0  # legs + waist
ARM_JOINT_COST = 0.3  # arms / wrists
R_WEIGHT = 1.0  # control effort (rad^2)
Q_REG = 1e-3  # regularization on all states

PUSH_BODY = "torso_link"
PUSH_DURATION = 0.1  # s
FALL_HEIGHT = 0.5  # pelvis z below this = fallen
KNEE_BEND = 0.3  # rad, standing knee bend


# ----------------------------------------------------------------------------
# Equilibrium
# ----------------------------------------------------------------------------
def set_leg_pose(model, data, hip):
    """Bent-knee stance with flat feet: hip + knee + ankle = 0 (sagittal)."""
    for side in ("left", "right"):
        data.joint(f"{side}_hip_pitch_joint").qpos = -hip
        data.joint(f"{side}_knee_joint").qpos = KNEE_BEND
        data.joint(f"{side}_ankle_pitch_joint").qpos = hip - KNEE_BEND
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    # Put the lowest foot contact point on the floor.
    foot_geoms = [g for g in range(model.ngeom) if model.geom_priority[g] == 1]
    lowest = (data.geom_xpos[foot_geoms, 2] - model.geom_size[foot_geoms, 0]).min()
    data.qpos[2] -= lowest
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def com_offset_x(model, data, hip):
    """CoM x minus support-centre x for a given hip/ankle split."""
    set_leg_pose(model, data, hip)
    _, sup = support_jacobian(model, data)
    return data.subtree_com[model.body("pelvis").id][0] - sup[0]


def find_standing_pose(model, data):
    """Bent-knee pose with the CoM over the middle of the feet, at the height
    where the feet exactly carry the robot's weight. Returns (qpos0, qfrc0)."""
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    hip = scipy.optimize.brentq(lambda h: com_offset_x(model, data, h), -0.6, 0.6)
    set_leg_pose(model, data, hip)
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    base_z = data.qpos[2]

    offsets = np.linspace(-0.01, 0.01, 2001)
    vertical = np.zeros_like(offsets)
    for i, off in enumerate(offsets):
        data.qpos[2] = base_z + off
        mujoco.mj_inverse(model, data)
        vertical[i] = data.qfrc_inverse[2]
    # Smallest residual vertical force = contacts support the weight.
    best = offsets[np.argmin(np.abs(vertical))]
    data.qpos[2] = base_z + best
    mujoco.mj_inverse(model, data)
    return data.qpos.copy(), data.qfrc_inverse.copy()


def actuator_dofs(model):
    """DoF index driven by each actuator (all G1 actuators are joint, gear 1)."""
    return model.jnt_dofadr[model.actuator_trnid[:, 0]]


def holding_ctrl(model, data, qpos0, qfrc0):
    """Position targets whose actuator forces reproduce qfrc0 at qpos0."""
    dofs = actuator_dofs(model)
    gear = model.actuator_gear[:, 0]
    force = qfrc0[dofs] / gear
    data.qpos[:] = qpos0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    # Position actuator: force = kp*ctrl + biasprm1*length + biasprm2*velocity
    kp = model.actuator_gainprm[:, 0]
    bias = model.actuator_biasprm[:, 1] * data.actuator_length
    return (force - bias) / kp


# ----------------------------------------------------------------------------
# LQR
# ----------------------------------------------------------------------------
def support_jacobian(model, data):
    """Jacobian (3 x nv) of the centre of the foot contact points."""
    jac = np.zeros((3, model.nv))
    jp = np.zeros((3, model.nv))
    foot_geoms = [g for g in range(model.ngeom) if model.geom_priority[g] == 1
                  and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    for g in foot_geoms:
        mujoco.mj_jac(model, data, jp, None, data.geom_xpos[g], model.geom_bodyid[g])
        jac += jp
    return jac / len(foot_geoms), data.geom_xpos[foot_geoms].mean(axis=0)


def build_lqr(model, data, qpos0, ctrl0):
    nv, nu = model.nv, model.nu
    data.qpos[:] = qpos0
    data.qvel[:] = 0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)

    # Balance cost: CoM minus support centre.
    jac_com = np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body("pelvis").id)
    jac_sup, _ = support_jacobian(model, data)
    jac_diff = jac_com - jac_sup
    q_balance = jac_diff.T @ jac_diff

    # Joint cost: none on the free root, more on legs/waist than arms.
    q_joint = np.zeros(nv)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(j).name
        is_arm = any(k in name for k in ("shoulder", "elbow", "wrist"))
        q_joint[model.jnt_dofadr[j]] = ARM_JOINT_COST if is_arm else LEG_JOINT_COST

    q_pos = BALANCE_COST * q_balance + np.diag(q_joint)
    Q = np.block([[q_pos, np.zeros((nv, nv))],
                  [np.zeros((nv, 2 * nv))]])
    # Small cost on every state so marginal modes (e.g. sliding the whole
    # robot sideways) are penalized and the Riccati equation is solvable.
    Q += Q_REG * np.eye(2 * nv)
    R = R_WEIGHT * np.eye(nu)

    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)

    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return dict(A=A, B=B, Q=Q, R=R, P=P, K=K, jac_balance=jac_diff,
                q_joint_diag=q_joint)


def save_lqr_data(model, ctrl, path):
    """Write the LQR design (matrices, weights, operating point) to .npz."""
    dof_names = ["base_x", "base_y", "base_z", "base_rx", "base_ry", "base_rz"]
    dof_names += [model.joint(j).name for j in range(1, model.njnt)]
    d = ctrl.lqr
    np.savez(
        path,
        # Linear model  dx[k+1] = A dx[k] + B du[k]
        A=d["A"], B=d["B"],
        # LQR  min sum dx'Q dx + du'R du,  du = -K dx
        Q=d["Q"], R=d["R"], P=d["P"], K=d["K"],
        # Pieces of Q
        jac_balance=d["jac_balance"], q_joint_diag=d["q_joint_diag"],
        balance_cost=BALANCE_COST, leg_joint_cost=LEG_JOINT_COST,
        arm_joint_cost=ARM_JOINT_COST, r_weight=R_WEIGHT, q_reg=Q_REG,
        # Operating point
        qpos0=ctrl.qpos0, ctrl0=ctrl.ctrl0, knee_bend=KNEE_BEND,
        dt=model.opt.timestep, fd_eps=1e-6,
        # Labels
        state_names=np.array([f"dq_{n}" for n in dof_names] + [f"v_{n}" for n in dof_names]),
        ctrl_names=np.array([model.actuator(i).name for i in range(model.nu)]),
        description=np.array(
            "Unitree G1 (mujoco_menagerie unitree_g1/scene.xml) standing-balance LQR. "
            "State dx = [dq (nv=35, mj_differentiatePos(qpos0, qpos)), qvel (35)]. "
            "Input du = ctrl - ctrl0, position-actuator joint targets [rad] (kp=500). "
            "A, B from mujoco.mjd_transitionFD (centred, eps=fd_eps) at (qpos0, ctrl0), "
            "discrete with step dt; contacts and actuator PD are inside A, B. "
            "Q = blkdiag(balance_cost*jac_balance'jac_balance + diag(q_joint_diag), 0) "
            "+ q_reg*I; R = r_weight*I; P from scipy solve_discrete_are; "
            "K = (R + B'PB)^-1 B'PA; control law ctrl = ctrl0 - K dx."),
    )


class Controller:
    def __init__(self, model, data):
        self.model = model
        self.qpos0, qfrc0 = find_standing_pose(model, data)
        self.ctrl0 = holding_ctrl(model, data, self.qpos0, qfrc0)
        self.lqr = build_lqr(model, data, self.qpos0, self.ctrl0)
        self.K = self.lqr["K"]
        self.dq = np.zeros(model.nv)
        self.use_lqr = True

    def reset(self, data):
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = self.qpos0
        data.ctrl[:] = self.ctrl0
        mujoco.mj_forward(self.model, data)

    def __call__(self, data):
        if not self.use_lqr:
            data.ctrl[:] = self.ctrl0
            return
        mujoco.mj_differentiatePos(self.model, self.dq, 1, self.qpos0, data.qpos)
        dx = np.hstack((self.dq, data.qvel))
        data.ctrl[:] = self.ctrl0 - self.K @ dx


# ----------------------------------------------------------------------------
# Pushes
# ----------------------------------------------------------------------------
class Pusher:
    def __init__(self, model, magnitude, period, rng, max_pushes=6):
        self.body = model.body(PUSH_BODY).id
        self.magnitude = magnitude
        self.period = period
        self.rng = rng
        self.force = np.zeros(3)
        self.t_start = -np.inf
        self.next_auto = 2.0
        self.max_pushes = max_pushes  # automatic pushes before stopping
        self.n_auto = 0

    def trigger(self, t, direction=None):
        if direction is None:
            ang = self.rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(ang), np.sin(ang), 0.0])
        self.force = self.magnitude * np.asarray(direction, float)
        self.t_start = t
        print(f"[t={t:6.2f}s] push {self.magnitude:.0f} N toward "
              f"({direction[0]:+.2f}, {direction[1]:+.2f})")

    def apply(self, data):
        t = data.time
        if (self.period > 0 and self.magnitude > 0 and t >= self.next_auto
                and self.n_auto < self.max_pushes):
            self.trigger(t)
            self.n_auto += 1
            self.next_auto = t + self.period
        active = t - self.t_start < PUSH_DURATION
        data.xfrc_applied[self.body, :3] = self.force if active else 0.0
        return active


def draw_overlay(viewer, model, data, pusher, active):
    """CoM projection (green), support centre (blue), push arrow (red)."""
    scn = viewer.user_scn
    scn.ngeom = 0

    def add(type_, size, pos, rgba, mat=np.eye(3)):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, type_, size, pos, mat.flatten(), rgba)
        scn.ngeom += 1
        return g

    com = data.subtree_com[model.body("pelvis").id].copy()
    _, sup = support_jacobian(model, data)
    add(mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
        np.array([com[0], com[1], 0.005]), np.array([0.1, 0.9, 0.1, 1]))
    add(mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0, 0]),
        np.array([sup[0], sup[1], 0.005]), np.array([0.2, 0.4, 1.0, 1]))

    if active:
        p = data.xpos[pusher.body]
        start = p - 0.5 * pusher.force / pusher.magnitude
        g = add(mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                np.array([1, 0.1, 0.1, 1]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, start, p)


# ----------------------------------------------------------------------------
# Main loops
# ----------------------------------------------------------------------------
def run_gui(args):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    ctrl = Controller(model, data)
    ctrl.reset(data)
    pusher = Pusher(model, args.push, args.period, np.random.default_rng(args.seed),
                    args.num_pushes)

    keys = {"push": False, "toggle": False, "reset": False}

    def on_key(code):
        if code == 260:  # Insert
            keys["push"] = True
        elif code == 269:  # End
            keys["toggle"] = True
        elif code == 268:  # Home
            keys["reset"] = True

    print(__doc__.split("GUI keys:")[1])
    print("Controller: LQR")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.7]
        viewer.cam.distance = 3.0
        while viewer.is_running():
            t0 = time.time()
            with viewer.lock():
                if keys["reset"]:
                    keys["reset"] = False
                    ctrl.reset(data)
                    pusher.next_auto = 2.0
                    pusher.n_auto = 0
                    pusher.rng = np.random.default_rng(args.seed)
                    print("reset")
                if keys["toggle"]:
                    keys["toggle"] = False
                    ctrl.use_lqr = not ctrl.use_lqr
                    print("Controller:", "LQR" if ctrl.use_lqr else "PD hold")
                if keys["push"]:
                    keys["push"] = False
                    pusher.trigger(data.time)
                # Step one frame worth of physics (~60 Hz render).
                n = max(1, int(round(1 / 60 / model.opt.timestep)))
                for _ in range(n):
                    active = pusher.apply(data)
                    ctrl(data)
                    mujoco.mj_step(model, data)
                draw_overlay(viewer, model, data, pusher, active)
            viewer.sync()
            dt = n * model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def survives(model, ctrl, use_lqr, force, direction, t_end=4.0):
    data = mujoco.MjData(model)
    ctrl.reset(data)
    ctrl.use_lqr = use_lqr
    body = model.body(PUSH_BODY).id
    while data.time < t_end:
        on = 1.0 <= data.time < 1.0 + PUSH_DURATION
        data.xfrc_applied[body, :3] = force * direction if on else 0.0
        ctrl(data)
        mujoco.mj_step(model, data)
        if data.qpos[2] < FALL_HEIGHT:
            return False
    return True


def run_test(args):
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    t0 = time.time()
    ctrl = Controller(model, data)
    print(f"LQR built in {time.time() - t0:.2f}s, K shape {ctrl.K.shape}")

    # Undisturbed: how far does the CoM drift?
    ctrl.reset(data)
    while data.time < 5.0:
        ctrl(data)
        mujoco.mj_step(model, data)
    print(f"undisturbed 5 s: pelvis z {data.qpos[2]:.3f} m, "
          f"xy drift {np.linalg.norm(data.qpos[:2] - ctrl.qpos0[:2])*1000:.1f} mm")

    # Largest 0.1 s torso push survived, per direction.
    dirs = {"+x (front)": [1, 0, 0], "-x (back)": [-1, 0, 0], "+y (side)": [0, 1, 0]}
    forces = np.arange(25, 1001, 25)
    print(f"\nmax survived 0.1 s push on {PUSH_BODY} [N]:")
    for name, d in dirs.items():
        d = np.array(d, float)
        row = []
        for use_lqr in (True, False):
            best = 0
            for f in forces:
                if survives(model, ctrl, use_lqr, f, d):
                    best = f
                else:
                    break
            row.append(best)
        print(f"  {name:11s}  LQR {row[0]:5.0f}   PD hold {row[1]:5.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--push", type=float, default=60.0, help="push force [N], 0 = off")
    p.add_argument("--period", type=float, default=4.0, help="auto push period [s], 0 = manual only")
    p.add_argument("--num-pushes", type=int, default=6,
                   help="automatic pushes before stopping (7th with seed 0 topples it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test", action="store_true", help="headless push test, no GUI")
    p.add_argument("--save", nargs="?", const=os.path.join(HERE, "g1_lqr_matrices.npz"),
                   help="save A, B, Q, R, P, K and operating point to .npz and exit")
    args = p.parse_args()
    if args.save:
        m = mujoco.MjModel.from_xml_path(SCENE)
        save_lqr_data(m, Controller(m, mujoco.MjData(m)), args.save)
        print("saved", args.save)
    elif args.test:
        run_test(args)
    else:
        run_gui(args)
