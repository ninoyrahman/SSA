"""Plotting helpers for propagated orbital states."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_locus_3d(t, r, v, *, interval=30, show=True):
    """Animate a 3D trajectory from time, position, and velocity arrays.

    Args:
        t: Time samples, shape ``(N,)``.
        r: Position samples, shape ``(N, 3)``.
        v: Velocity samples, shape ``(N, 3)``. Speed is shown in the title.
        interval: Delay between animation frames, in milliseconds.
        show: Whether to display the animation using Matplotlib.

    Returns:
        ``(fig, ax, animation)`` so callers can customize or retain the plot.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t must be a non-empty 1D array")
    if r.shape != (t.size, 3) or v.shape != (t.size, 3):
        raise ValueError("r and v must each have shape (len(t), 3)")
    if not (np.isfinite(t).all() and np.isfinite(r).all() and np.isfinite(v).all()):
        raise ValueError("t, r, and v must contain only finite values")
    if interval <= 0:
        raise ValueError("interval must be positive")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    equatorial_radius = max(float(np.max(np.linalg.norm(r, axis=1))), 1e-6)
    view_radius = max(float(np.max(np.abs(r))), equatorial_radius) * 1.1
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.set_zlim(-view_radius, view_radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    equatorial_angle = np.linspace(0.0, 2 * np.pi, 361)
    equatorial_x = equatorial_radius * np.cos(equatorial_angle)
    equatorial_y = equatorial_radius * np.sin(equatorial_angle)
    ax.plot(
        equatorial_x,
        equatorial_y,
        np.zeros_like(equatorial_angle),
        color="red",
        lw=1.5,
        label="Equatorial geodesic (z=0)",
    )

    line, = ax.plot([], [], [], lw=2, label="Trajectory")
    point, = ax.plot([], [], [], "ro", markersize=6, label="Spacecraft")
    ax.legend()

    def update(i):
        line.set_data(r[:i + 1, 0], r[:i + 1, 1])
        line.set_3d_properties(r[:i + 1, 2])
        point.set_data([r[i, 0]], [r[i, 1]])
        point.set_3d_properties([r[i, 2]])
        speed = np.linalg.norm(v[i])
        ax.set_title(f"t = {t[i]:.4f}, speed = {speed:.4f}")
        return line, point

    animation = FuncAnimation(
        fig, update, frames=t.size, interval=interval, blit=False
    )
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        save_path = filedialog.asksaveasfilename(
            parent=root,
            title="Save orbit animation",
            defaultextension=".gif",
            filetypes=(("GIF animation", "*.gif"),),
        )
    finally:
        root.destroy()

    if save_path:
        if not str(save_path).lower().endswith(".gif"):
            save_path = f"{save_path}.gif"
        animation.save(
            save_path, writer=PillowWriter(fps=max(1, 1000 // interval))
        )
    if show:
        plt.show()
    return fig, ax, animation
