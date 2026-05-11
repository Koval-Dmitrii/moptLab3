from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _plot_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return RESULTS_DIR / "plots" / path


def _table_path(title: str) -> Path:
    safe_title = "".join(c if (c.isalnum() or c in (" ", "_")) else "_" for c in title or "table")
    return RESULTS_DIR / "tables" / f"{safe_title.strip().replace(' ', '_')}.csv"


def plot_contour_with_trajectory(
    f, title, trajectories: dict,
    xlim=(-3, 3), ylim=(-3, 3),
    n_levels=30, figsize=(8, 6), filename=None, save=True
):
    xx, yy = np.meshgrid(
        np.linspace(*xlim, 350),
        np.linspace(*ylim, 350)
    )
    zz = np.array([[f(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
                   for row_x, row_y in zip(xx, yy)])

    fig, ax = plt.subplots(figsize=figsize)
    finite = zz[np.isfinite(zz)]
    levels = np.percentile(finite, np.linspace(0, 95, n_levels))
    cs = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", alpha=0.7)
    ax.contour(xx, yy, zz, levels=levels, colors="white", linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(trajectories), 1)))
    for (label, traj), color in zip(trajectories.items(), colors):
        pts = np.array(traj)
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, label=label, markersize=2, linewidth=1.2)
            ax.plot(pts[0, 0], pts[0, 1], "s", color=color, markersize=5)
            ax.plot(pts[-1, 0], pts[-1, 1], "*", color=color, markersize=7)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    if save and filename:
        save_path = _plot_path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_iter_vs_step(steps, iters, title, xlabel="Параметр", filename=None, save=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, iters, "o-", color="steelblue", linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Число итераций (log)")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save and filename:
        save_path = _plot_path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_heatmap(values, x_values, y_values, title, xlabel, ylabel, filename=None, save=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Итерации")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_xticklabels([f"{v:g}" for v in x_values], rotation=45, ha="right")
    ax.set_yticklabels([f"{v:g}" for v in y_values])

    for yi in range(values.shape[0]):
        for xi in range(values.shape[1]):
            val = values[yi, xi]
            txt = "—" if not np.isfinite(val) else str(int(val))
            ax.text(xi, yi, txt, ha="center", va="center", color="white", fontsize=7)

    plt.tight_layout()
    if save and filename:
        save_path = _plot_path(filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_table(headers, rows, title="", save=True):
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
             for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  ".join("─" * w for w in col_w))
    for row in rows:
        print(fmt.format(*row))
    print()

    if save:
        save_path = _table_path(title)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow([str(item) for item in r])
