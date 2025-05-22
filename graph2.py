# visual 3d graph ver 
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
G = 9.81
RHO = 1.225
AIR_VISCOSITY = 1.516e-05
BALL_MASS = 0.0027
BALL_DIAMETER = 0.04
BALL_RADIUS = BALL_DIAMETER / 2.0
BALL_AREA = math.pi * BALL_RADIUS ** 2
BALL_INERTIA = 2.0 / 5.0 * BALL_MASS * BALL_RADIUS ** 2
DRAG_COEFF_BASE = 0.47
TIME_STEP = 0.0005
MAX_STEPS = 30000

def calculate_reynolds_number(velocity_magnitude, diameter, viscosity):
    if viscosity < 1e-09:
        return 0.0
    return max(1e-06, velocity_magnitude * diameter / viscosity)

def calculate_drag_coefficient(reynolds_number):
    if reynolds_number < 1.0:
        cd = 24.0 / reynolds_number
    elif reynolds_number < 1000.0:
        cd = 24.0 / reynolds_number * (1.0 + 0.15 * reynolds_number ** 0.687)
    elif reynolds_number < 200000.0:
        cd = 0.47
    elif reynolds_number < 350000.0:
        cd = 0.47 - (0.47 - 0.1) * (reynolds_number - 200000.0) / (350000.0 - 200000.0)
    else:
        cd = 0.15
    return max(cd, 0.05)

def calculate_lift_coefficient(spin_rpm, velocity_magnitude, ball_radius, reynolds_number):
    if velocity_magnitude < 1e-06:
        return 0.0
    omega_rad_s = spin_rpm * (2 * math.pi) / 60.0
    spin_parameter = omega_rad_s * ball_radius / velocity_magnitude
    cl_base = min(abs(spin_parameter), 0.6) * np.sign(spin_parameter)
    reduction_factor = max(0.5, 1.0 - reynolds_number / 1000000.0)
    cl = cl_base * reduction_factor
    return cl

def calculate_spin_decay_torque(omega_rad_s, velocity_magnitude):
    C_damp = 5e-09
    torque = -C_damp * omega_rad_s * velocity_magnitude ** 1.5
    return torque

def calculate_forces_detailed(state):
    (x, y, z, vx, vy, vz, omega_z) = state
    v_mag_sq = vx ** 2 + vy ** 2 + vz ** 2
    v_mag = math.sqrt(v_mag_sq)
    Re = calculate_reynolds_number(v_mag, BALL_DIAMETER, AIR_VISCOSITY)
    Fg_y = -BALL_MASS * G
    Cd = calculate_drag_coefficient(Re)
    Fd_mag = 0.5 * RHO * v_mag_sq * Cd * BALL_AREA
    if v_mag > 1e-06:
        Fd_x = -Fd_mag * (vx / v_mag)
        Fd_y = -Fd_mag * (vy / v_mag)
        Fd_z = -Fd_mag * (vz / v_mag)
    else:
        (Fd_x, Fd_y, Fd_z) = (0.0, 0.0, 0.0)
    spin_rpm_current = omega_z * 60.0 / (2 * math.pi)
    Cl = calculate_lift_coefficient(spin_rpm_current, v_mag, BALL_RADIUS, Re)
    Fm_mag = 0.5 * RHO * v_mag_sq * Cl * BALL_AREA
    spin_axis_norm = np.array([0.0, 0.0, 1.0])
    vel_vec = np.array([vx, vy, vz])
    spin_vec = omega_z * spin_axis_norm
    (Fm_x, Fm_y, Fm_z) = (0.0, 0.0, 0.0)
    if v_mag > 1e-06 and abs(omega_z) > 1e-06:
        magnus_force_vec_dir = np.cross(spin_vec, vel_vec)
        norm_magnus_dir = np.linalg.norm(magnus_force_vec_dir)
        if norm_magnus_dir > 1e-06:
            magnus_force_vec = magnus_force_vec_dir / norm_magnus_dir * Fm_mag
            (Fm_x, Fm_y, Fm_z) = (magnus_force_vec[0], magnus_force_vec[1], magnus_force_vec[2])
    F_total_x = Fd_x + Fm_x
    F_total_y = Fg_y + Fd_y + Fm_y
    F_total_z = Fd_z + Fm_z
    spin_decay_torque_z = calculate_spin_decay_torque(omega_z, v_mag)
    alpha_z = spin_decay_torque_z / BALL_INERTIA
    return (F_total_x, F_total_y, F_total_z, alpha_z)

def derivatives_detailed(state, t):
    (x, y, z, vx, vy, vz, omega_z) = state
    (Fx, Fy, Fz, alpha_z) = calculate_forces_detailed(state)
    ax = Fx / BALL_MASS
    ay = Fy / BALL_MASS
    az = Fz / BALL_MASS
    return np.array([vx, vy, vz, ax, ay, az, alpha_z])

def rk4_step_detailed(state, t, dt):
    k1 = dt * derivatives_detailed(state, t)
    k2 = dt * derivatives_detailed(state + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * derivatives_detailed(state + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * derivatives_detailed(state + k3, t + dt)
    state_new = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return state_new

def simulate_trajectory_rk4_detailed(v0, angle_deg, y0, initial_spin_rpm, spin_axis_norm):
    if v0 <= 0:
        return ([], 0, 0, 0)
    angle_rad = math.radians(angle_deg)
    vx0 = v0 * math.cos(angle_rad)
    vy0 = v0 * math.sin(angle_rad)
    vz0 = 0.0
    omega_z0 = initial_spin_rpm * (2 * math.pi) / 60.0
    state = np.array([0.0, y0, 0.0, vx0, vy0, vz0, omega_z0])
    t = 0.0
    max_height = y0
    trajectory = [state[:3].copy()]
    step = 0
    while state[1] >= -0.01 and step < MAX_STEPS:
        state_prev = state.copy()
        state = rk4_step_detailed(state, t, TIME_STEP)
        t += TIME_STEP
        step += 1
        if state[1] > max_height:
            max_height = state[1]
        trajectory.append(state[:3].copy())
    if step >= MAX_STEPS:
        print(f'\nWarning: Simulation exceeded max steps for angle {angle_deg} deg, spin {initial_spin_rpm} RPM.')
        return (trajectory, -1, -1, -1)
    (range_m, flight_time) = (0, 0)
    if state[1] < 0 and step > 1:
        (y_prev, y_curr) = (state_prev[1], state[1])
        (x_prev, x_curr) = (state_prev[0], state[0])
        (z_prev, z_curr) = (state_prev[2], state[2])
        t_prev = t - TIME_STEP
        if abs(y_curr - y_prev) > 1e-09:
            interp_factor = y_prev / (y_prev - y_curr)
        else:
            interp_factor = 0
        final_x = x_prev + (x_curr - x_prev) * interp_factor
        final_z = z_prev + (z_curr - z_prev) * interp_factor
        final_t = t_prev + TIME_STEP * interp_factor
        range_m = math.sqrt(final_x ** 2 + final_z ** 2)
        flight_time = final_t
        trajectory.append(np.array([final_x, 0.0, final_z]))
    elif state[1] >= 0:
        range_m = math.sqrt(state[0] ** 2 + state[2] ** 2)
        flight_time = t
    return (np.array(trajectory), range_m, max_height, flight_time)

def precompute_ranges(v0, y0, initial_spin_rpm, spin_axis_norm, angles):
    print('Precomputing ranges...')
    ranges = {}
    trajectories = {}
    max_heights = {}
    flight_times = {}
    for angle in angles:
        print(f'  Simulating {angle:.1f}°...')
        (traj, r, h, t) = simulate_trajectory_rk4_detailed(v0, angle, y0, initial_spin_rpm, spin_axis_norm)
        if r >= 0:
            ranges[angle] = r
            trajectories[angle] = traj
            max_heights[angle] = h
            flight_times[angle] = t
        else:
            print(f'    Simulation failed for angle {angle:.1f} - excluding.')
            ranges[angle] = 0
            trajectories[angle] = np.array([[0, y0, 0]])
            max_heights[angle] = 0
            flight_times[angle] = 0
    print('Precomputation complete.')
    return (ranges, trajectories, max_heights, flight_times)
if __name__ == '__main__':
    initial_velocity_mps = 8.0
    launch_height_m = 0.15
    initial_spin_rpm = 2000
    spin_axis = np.array([0.0, 0.0, 1.0])
    min_launch_angle_deg = 0
    max_launch_angle_deg = 30
    angle_step_deg = 1.0
    angles_to_sim = np.arange(min_launch_angle_deg, max_launch_angle_deg + angle_step_deg, angle_step_deg)
    (sim_ranges, sim_trajectories, sim_max_heights, sim_flight_times) = precompute_ranges(initial_velocity_mps, launch_height_m, initial_spin_rpm, spin_axis, angles_to_sim)
    valid_angles = [a for a in angles_to_sim if sim_ranges.get(a, 0) > 0]
    if not valid_angles:
        print('\nERROR: All simulations failed. Check parameters.')
        exit()
    optimal_angle = max(valid_angles, key=lambda angle: sim_ranges.get(angle, 0))
    max_distance = sim_ranges.get(optimal_angle, 0)
    max_h_at_opt = sim_max_heights.get(optimal_angle, 0)
    time_at_opt = sim_flight_times.get(optimal_angle, 0)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    ax_traj = fig.add_subplot(gs[0, 0], projection='3d')
    ax_range = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')
    angles_plot = sorted(valid_angles)
    ranges_plot = [sim_ranges.get(a, 0) for a in angles_plot]
    ax_range.plot(angles_plot, ranges_plot, marker='o', linestyle='-', color='royalblue')
    (opt_marker,) = ax_range.plot(optimal_angle, max_distance, 'ro', markersize=10, label=f'Simulated Optimum ({optimal_angle:.1f}°)')
    ax_range.set_xlabel('Launch Angle (degrees)')
    ax_range.set_ylabel('Horizontal Range (m)')
    ax_range.set_title('Simulated Range vs. Launch Angle')
    ax_range.legend()
    ax_range.grid(True)
    initial_traj = sim_trajectories.get(optimal_angle, np.array([[0, launch_height_m, 0]]))
    if initial_traj.shape[1] == 3:
        (line_traj,) = ax_traj.plot(initial_traj[:, 0], initial_traj[:, 2], initial_traj[:, 1], lw=2, color='darkorange', label=f'{optimal_angle:.1f}° Trajectory')
        ax_traj.set_xlabel('X Distance (m)')
        ax_traj.set_ylabel('Z Distance (m)')
        ax_traj.set_zlabel('Y Height (m)')
        ax_traj.set_title('Simulated 3D Trajectory')
        max_x_lim = max(1, np.max(initial_traj[:, 0]) * 1.1) if initial_traj.size > 0 else 1
        max_z_lim = max(0.5, np.max(np.abs(initial_traj[:, 2])) * 1.2) if initial_traj.size > 0 else 0.5
        max_y_lim = max(0.5, np.max(initial_traj[:, 1]) * 1.1) if initial_traj.size > 0 else 0.5
        ax_traj.set_xlim(0, max_x_lim)
        ax_traj.set_ylim(-max_z_lim, max_z_lim)
        ax_traj.set_zlim(0, max_y_lim)
    else:
        (line_traj,) = ax_traj.plot(initial_traj[:, 0], initial_traj[:, 1], lw=2, color='darkorange', label=f'{optimal_angle:.1f}° Trajectory')
        ax_traj.set_xlabel('X Distance (m)')
        ax_traj.set_ylabel('Y Height (m)')
        ax_traj.set_title('Simulated 2D Trajectory (X-Y)')
        ax_traj.axis('equal')
    ax_traj.legend(loc='upper right')
    ax_traj.grid(True)
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9))

    def update_info_text(angle):
        r = sim_ranges.get(angle, 0)
        h = sim_max_heights.get(angle, 0)
        t = sim_flight_times.get(angle, 0)
        text = f'Selected Angle: {angle:.1f}°\n'
        text += f'Simulated Range: {r:.2f} m\n'
        text += f'Simulated Max Height: {h:.2f} m\n'
        text += f'Simulated Flight Time: {t:.2f} s\n\n'
        text += f'Overall Optimal Angle ({min_launch_angle_deg}-{max_launch_angle_deg}°): {optimal_angle:.1f}°\n'
        text += f'Max Simulated Range: {max_distance:.2f} m\n\n'
        text += f'Inputs:\n'
        text += f'  v0 = {initial_velocity_mps:.1f} m/s (Est.)\n'
        text += f'  y0 = {launch_height_m:.2f} m\n'
        text += f'  Spin = {initial_spin_rpm} RPM\n'
        text += f'  Spin Axis = {spin_axis}\n'
        text += f'  Cd Model = Variable(Re)\n'
        text += f'  Magnus Model = Simple(S, Re)\n'
        text += f'  Spin Decay = Simple Model'
        info_text.set_text(text)
    update_info_text(optimal_angle)
    fig.subplots_adjust(bottom=0.25, hspace=0.4)
    ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    angle_slider = Slider(ax=ax_slider, label='Launch Angle (°)', valmin=min_launch_angle_deg, valmax=max_launch_angle_deg, valinit=optimal_angle, valstep=angle_step_deg, color='lightblue')

    def update(val):
        selected_angle = angle_slider.val
        closest_angle = min(angles_to_sim, key=lambda x: abs(x - selected_angle))
        new_traj = sim_trajectories.get(closest_angle, np.array([[0, launch_height_m, 0]]))
        if new_traj.shape[1] == 3:
            line_traj.set_data_3d(new_traj[:, 0], new_traj[:, 2], new_traj[:, 1])
            max_x_lim = max(1, np.max(new_traj[:, 0]) * 1.1) if new_traj.size > 0 else 1
            max_z_lim = max(0.5, np.max(np.abs(new_traj[:, 2])) * 1.2) if new_traj.size > 0 else 0.5
            max_y_lim = max(0.5, np.max(new_traj[:, 1]) * 1.1) if new_traj.size > 0 else 0.5
            ax_traj.set_xlim(0, max_x_lim)
            ax_traj.set_ylim(-max_z_lim, max_z_lim)
            ax_traj.set_zlim(0, max_y_lim)
        else:
            line_traj.set_data(new_traj[:, 0], new_traj[:, 1])
            ax_traj.relim()
            ax_traj.autoscale_view(True, True, True)
        line_traj.set_label(f'{closest_angle:.1f}° Trajectory')
        ax_traj.legend(loc='upper right')
        update_info_text(closest_angle)
        fig.canvas.draw_idle()
    angle_slider.on_changed(update)
    plt.show()
