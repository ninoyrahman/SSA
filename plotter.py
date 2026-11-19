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
    low = np.min(r, axis=0)
    high = np.max(r, axis=0)
    center = (low + high) / 2
    radius = max(float(np.max(high - low)) / 2, 1e-6) * 1.1
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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
