import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

G = 9.81
RHO = 1.225
VISC = 1.516e-5
MASS = 0.0027
DIAM = 0.04
RADIUS = DIAM / 2
AREA = math.pi * RADIUS ** 2
INERTIA = (2/5) * MASS * RADIUS ** 2
TIME_STEP = 0.0005
MAX_STEPS = 30000


def reynolds_number(v, d, nu):
    return max(1e-6, (v * d) / nu) if nu >= 1e-9 else 0.0


def drag_coefficient(Re):
    if Re < 1.0:
        return 24.0 / Re
    if Re < 1e3:
        return 24.0 / Re * (1 + 0.15 * Re ** 0.687)
    if Re < 2e5:
        return 0.47
    if Re < 3.5e5:
        return 0.47 - (0.37 * (Re - 2e5)) / 1.5e5
    return 0.15


def lift_coefficient(spin_rpm, v, r, Re):
    if v < 1e-6:
        return 0.0
    omega = spin_rpm * 2 * math.pi / 60
    s = (omega * r) / v
    cl = min(abs(s), 0.6) * np.sign(s)
    return cl * max(0.5, 1 - Re / 1e6)


def spin_decay_torque(omega, v):
    return -5e-9 * omega * v ** 1.5


def compute_forces(state):
    x, y, z, vx, vy, vz, omega_z = state
    v_mag = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    Re = reynolds_number(v_mag, DIAM, VISC)
    Cd = drag_coefficient(Re)
    Cl = lift_coefficient(omega_z * 60 / (2 * math.pi), v_mag, RADIUS, Re)
    Fd = 0.5 * RHO * v_mag ** 2 * Cd * AREA
    Fm = 0.5 * RHO * v_mag ** 2 * Cl * AREA

    fx_drag = -Fd * vx / v_mag if v_mag > 1e-6 else 0
    fy_drag = -Fd * vy / v_mag if v_mag > 1e-6 else 0
    fz_drag = -Fd * vz / v_mag if v_mag > 1e-6 else 0

    vel_vec = np.array([vx, vy, vz])
    spin_vec = omega_z * np.array([0, 0, 1])
    mag_force = np.cross(spin_vec, vel_vec)
    mag_dir = np.linalg.norm(mag_force)
    mag_force = (mag_force / mag_dir) * Fm if mag_dir > 1e-6 else np.zeros(3)

    alpha_z = spin_decay_torque(omega_z, v_mag) / INERTIA

    fx, fy, fz = fx_drag + mag_force[0], -MASS * G + fy_drag + mag_force[1], fz_drag + mag_force[2]
    return fx, fy, fz, alpha_z


def derivatives(state, t):
    x, y, z, vx, vy, vz, omega_z = state
    fx, fy, fz, alpha_z = compute_forces(state)
    return np.array([vx, vy, vz, fx / MASS, fy / MASS, fz / MASS, alpha_z])


def rk4_step(state, t, dt):
    k1 = dt * derivatives(state, t)
    k2 = dt * derivatives(state + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * derivatives(state + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives(state + k3, t + dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def simulate(v0, angle_deg, y0, spin_rpm):
    if v0 <= 0:
        return [], 0, 0, 0
    angle_rad = math.radians(angle_deg)
    state = np.array([0, y0, 0, v0 * math.cos(angle_rad), v0 * math.sin(angle_rad), 0, spin_rpm * 2 * math.pi / 60])
    t = 0
    step = 0
    max_y = y0
    traj = [state[:3].copy()]

    while state[1] >= -0.01 and step < MAX_STEPS:
        prev = state.copy()
        state = rk4_step(state, t, TIME_STEP)
        t += TIME_STEP
        step += 1
        max_y = max(max_y, state[1])
        traj.append(state[:3].copy())

    if step >= MAX_STEPS:
        return traj, -1, -1, -1

    if state[1] < 0:
        dy = prev[1] - state[1]
        interp = prev[1] / dy if abs(dy) > 1e-9 else 0
        final_x = prev[0] + (state[0] - prev[0]) * interp
        final_z = prev[2] + (state[2] - prev[2]) * interp
        final_t = t - TIME_STEP + TIME_STEP * interp
        dist = math.sqrt(final_x ** 2 + final_z ** 2)
        traj.append(np.array([final_x, 0, final_z]))
        return np.array(traj), dist, max_y, final_t

    dist = math.sqrt(state[0] ** 2 + state[2] ** 2)
    return np.array(traj), dist, max_y, t


def batch_simulate(v0, y0, spin_rpm, angles):
    results = {}
    for a in angles:
        traj, dist, height, duration = simulate(v0, a, y0, spin_rpm)
        results[a] = {
            'trajectory': traj,
            'range': dist,
            'max_height': height,
            'flight_time': duration
        }
    return results


if __name__ == "__main__":
    angles = np.arange(0, 31, 1)
    sim_data = batch_simulate(8.0, 0.15, 2000, angles)

    best_angle = max(sim_data, key=lambda k: sim_data[k]['range'])
    best_traj = sim_data[best_angle]['trajectory']

    plt.figure(figsize=(10, 6))
    for a in sim_data:
        t = sim_data[a]['trajectory']
        if len(t):
            plt.plot(t[:, 0], t[:, 1], label=f"{a:.0f}°")
    plt.plot(best_traj[:, 0], best_traj[:, 1], 'k-', lw=2, label=f"Optimal: {best_angle}°")
    plt.title("Trajectories for Varying Launch Angles")
    plt.xlabel("X Distance (m)")
    plt.ylabel("Y Height (m)")
    plt.grid(True)
    plt.legend()
    plt.show()
