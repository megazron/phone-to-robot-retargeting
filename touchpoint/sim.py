"""Tabletop pick-and-place scene: Franka Panda (mujoco_menagerie), a disc and a mug.

Control interface is Cartesian: each step takes a target TCP position and a binary
gripper command. A damped-least-squares IK turns that into joint position targets for
the Panda's built-in position actuators, so grasping is fully physical (no welds).
"""
import os
from pathlib import Path

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"   # headless default; set MUJOCO_GL=egl on GPU servers
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANDA_DIR = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"

CTRL_DT = 0.05            # 20 Hz policy / replay rate
CUBE_HALF = 0.015         # 4.2 x 4.2 x 3 cm block standing in for the ~4 cm plastic disc
MUG_R = 0.05            # inner radius
MUG_H = 0.10
TCP_OFFSET = np.array([0.0, 0.0, 0.1034])   # hand frame -> fingertip centre
FLOOR_Z = 0.008
WS_LO = np.array([0.2, -0.5, FLOOR_Z])
WS_HI = np.array([0.8, 0.5, 0.55])
MAX_STEP = 0.025          # max TCP reference displacement per control step (m)

# reachable layout region (robot base at origin, table = floor plane)
X_RANGE = (0.35, 0.62)
Y_RANGE = (-0.28, 0.28)
MIN_SEP = 0.18

SCENE_XML = """
<mujoco model="touchpoint_tabletop">
  <include file="panda.xml"/>
  <option timestep="0.002" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <texture type="2d" name="table" builtin="checker" mark="edge" rgb1="0.78 0.72 0.62"
      rgb2="0.74 0.68 0.58" markrgb="0.6 0.55 0.48" width="300" height="300"/>
    <material name="table" texture="table" texuniform="true" texrepeat="6 6" reflectance="0.05"/>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.4 0.45" rgb2="0.1 0.1 0.12" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.3 1.6" dir="0 0.2 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="table" size="0 0 0.05" type="plane" material="table"/>
    <body name="cube" pos="0.5 0 0.01">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="0.021 0.021 0.015" mass="0.02" friction="2.5 0.05 0.001"
        condim="4" rgba="0.95 0.95 0.93 1"/>
    </body>
    <body name="mug" mocap="true" pos="0.5 0.2 0">
      <geom type="cylinder" size="0.056 0.004" pos="0 0 0.004" rgba="0.08 0.08 0.09 1"/>
      MUG_WALLS
    </body>
  </worldbody>
</mujoco>
"""


def _mug_walls(n=18, r=0.053, h=0.05):
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n
        half_len = r * np.tan(np.pi / n) + 0.002
        out.append(
            f'<geom type="box" size="0.004 {half_len:.4f} {h:.3f}" pos="{r*np.cos(a):.4f} {r*np.sin(a):.4f} {h:.3f}" '
            f'euler="0 0 {a:.4f}" rgba="0.08 0.08 0.09 1"/>'
        )
    return "\n      ".join(out)


def scene_path():
    if not (PANDA_DIR / "panda.xml").exists():
        raise FileNotFoundError(
            f"Panda model not found at {PANDA_DIR}. Run `make setup` (scripts/setup.sh) first."
        )
    p = PANDA_DIR / "touchpoint_scene.xml"
    xml = SCENE_XML.replace("MUG_WALLS", _mug_walls())
    if not p.exists() or p.read_text() != xml:
        p.write_text(xml)
    return p


def sample_layout(rng):
    """Random cube / mug xy positions with a minimum separation."""
    while True:
        c = rng.uniform([X_RANGE[0], Y_RANGE[0]], [X_RANGE[1], Y_RANGE[1]])
        b = rng.uniform([X_RANGE[0], Y_RANGE[0]], [X_RANGE[1], Y_RANGE[1]])
        if np.linalg.norm(c - b) > MIN_SEP:
            return {"cube": c, "mug": b}


class PickPlaceEnv:
    def __init__(self, render_size=None):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path()))
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)
        self.n_sub = int(round(CTRL_DT / self.model.opt.timestep))
        self.hand_id = self.model.body("hand").id
        self.cube_jnt = self.model.joint("cube").qposadr[0]
        self.cube_dof = self.model.joint("cube").dofadr[0]
        self.mug_mocap = self.model.body("mug").mocapid[0]
        self.home_q = self.model.key("home").qpos[:7].copy()
        self.q_lo = self.model.jnt_range[:7, 0]
        self.q_hi = self.model.jnt_range[:7, 1]
        self.renderer = None
        if render_size:
            self.renderer = mujoco.Renderer(self.model, height=render_size[1], width=render_size[0])
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.42, 0.0, 0.22]
            self.cam.distance = 1.6
            self.cam.azimuth = 200
            self.cam.elevation = -24

    # ---------- state ----------
    def reset(self, layout):
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[self.cube_jnt:self.cube_jnt + 3] = [*layout["cube"], CUBE_HALF]
        d.qpos[self.cube_jnt + 3:self.cube_jnt + 7] = [1, 0, 0, 0]
        d.mocap_pos[self.mug_mocap] = [*layout["mug"], 0.0]
        mujoco.mj_forward(m, d)
        self.q_des = self.home_q.copy()
        self.grip = 0.0
        self.target_quat = d.xquat[self.hand_id].copy()
        self.home_tcp = self.tcp().copy()
        self.ref = self.home_tcp.copy()
        # let things settle
        for _ in range(10):
            self.step(self.home_tcp, 0.0)
        return self.obs()

    def tcp(self, d=None):
        d = self.data if d is None else d
        return d.xpos[self.hand_id] + d.xmat[self.hand_id].reshape(3, 3) @ TCP_OFFSET

    def cube_pos(self):
        return self.data.qpos[self.cube_jnt:self.cube_jnt + 3].copy()

    def mug_pos(self):
        return self.data.mocap_pos[self.mug_mocap].copy()

    def finger_width(self):
        return float(self.data.qpos[7] + self.data.qpos[8])

    def obs(self):
        return {
            "tcp": self.tcp().copy(),
            "ref": self.ref.copy(),
            "cube": self.cube_pos(),
            "mug": self.mug_pos(),
            "width": self.finger_width(),
            "grip": self.grip,
        }

    def nudge_cube(self, dxy):
        self.data.qpos[self.cube_jnt:self.cube_jnt + 2] += dxy
        self.data.qvel[self.cube_dof:self.cube_dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def success(self):
        c = self.cube_pos()
        in_mug = np.linalg.norm(c[:2] - self.mug_pos()[:2]) < MUG_R and c[2] < MUG_H - 0.01
        return bool(in_mug and self.grip < 0.5 and self.finger_width() > 0.05)

    # ---------- control ----------
    def _solve_ik(self, target, iters=6, damping=0.05):
        m, d = self.model, self._ik
        q = self.q_des.copy()
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        err = np.zeros(6)
        qdiff = np.zeros(3)
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            tcp = self.tcp(d)
            err[:3] = target - tcp
            mujoco.mju_subQuat(qdiff, self.target_quat, d.xquat[self.hand_id])
            err[3:] = d.xmat[self.hand_id].reshape(3, 3) @ qdiff
            mujoco.mj_jac(m, d, jp, jr, tcp, self.hand_id)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            # weak pull to home posture in the nullspace
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (self.home_q - q))
            q = np.clip(q + dq, self.q_lo, self.q_hi)
        return q

    def step(self, target_tcp, grip):
        """target_tcp: desired TCP position (clipped to MAX_STEP from current); grip: 1 = close."""
        # the commanded reference moves at most MAX_STEP per step; the servos follow it
        delta = np.asarray(target_tcp, float) - self.ref
        n = np.linalg.norm(delta)
        if n > MAX_STEP:
            delta *= MAX_STEP / n
        self.ref = self.ref + delta
        self.ref = np.clip(self.ref, WS_LO, WS_HI)   # workspace box; z floor keeps fingertips off the table
        self.q_des = self._solve_ik(self.ref)
        self.grip = float(grip > 0.5)
        self.data.ctrl[:7] = self.q_des
        self.data.ctrl[7] = 0.0 if self.grip else 255.0
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        return self.obs()

    def render(self):
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()
