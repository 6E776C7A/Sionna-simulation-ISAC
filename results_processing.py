from __future__ import annotations

import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "wyniki"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def finite(value: str) -> float | None:
    if value in ("", "None", "nan", None):
        return None
    return float(value)


def normalize_rows(path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    # Sprawdzamy czy plik używa frame_id czy proper_frame_id
    frame_key = "frame_id" if "frame_id" in rows[0] else "proper_frame_id"
    offset = 1 if frame_key == "frame_id" else 0
    return {int(row[frame_key]) + offset: row for row in rows}


def value(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            return finite(row[key])
    return None


def display_label(path: Path) -> str:
    try:
        relative = path.relative_to(RESULTS_ROOT)
        filename = relative.name.removeprefix("simulation_data_")
        parent_parts = relative.parts[:-1]
        if len(parent_parts) >= 2 and parent_parts[0] == "random_seeds":
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f"{stem.replace('_results', '')}{parent_parts[1]}_results{suffix}"
            parent_parts = parent_parts[:2]
        elif len(parent_parts) > 1:
            parent_parts = parent_parts[:1]
        return Path(*parent_parts, filename).as_posix()
    except ValueError:
        return path.name


def compare_to_base(
    base: dict[int, dict[str, str]],
    candidate: dict[int, dict[str, str]],
) -> dict[str, object]:
    fields = {
        "range_m": ("range_m",),
        "velocity_m_s": ("velocity_m_s", "radial_velocity_m_s"),
        "azimuth_deg": ("music_azimuth_deg", "azimuth_local_deg"),
        "elevation_deg": ("music_elevation_deg", "elevation_local_deg"),
        "kalman_velocity_m_s": ("kalman_velocity_m_s", "ref_velocity_m_s"),
        "kalman_x": ("kalman_x", "ref_x"),
        "kalman_y": ("kalman_y", "ref_y"),
        "kalman_z": ("kalman_z", "ref_z"),
    }
    errors = {name: [] for name in fields}
    common = sorted(set(base) & set(candidate))
    for frame in common:
        for name, keys in fields.items():
            left = value(base[frame], *keys)
            right = value(candidate[frame], *keys)
            if left is not None and right is not None:
                errors[name].append(right - left)
    metrics: dict[str, object] = {"common_frames": len(common)}
    for name, values in errors.items():
        metrics[f"{name}_mae"] = np.mean(np.abs(values)) if values else None
        metrics[f"{name}_bias"] = np.mean(values) if values else None
    return metrics


def plot_cfar_music_comparisons(
    datasets: list[tuple[str, dict[int, dict[str, str]]]],
) -> None:
    """Wykres 1: Podstawowe parametry z CFAR i MUSIC (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("CFAR & MUSIC Measurements Comparison", fontsize=14, fontweight="bold")
    series = [
        (("range_m",), "Range [m]"),
        (("velocity_m_s", "radial_velocity_m_s"), "Velocity [m/s]"),
        (("music_azimuth_deg", "azimuth_local_deg"), "Azimuth [deg]"),
        (("music_elevation_deg", "elevation_local_deg"), "Elevation [deg]"),
    ]

    for axis, (keys, ylabel) in zip(axes.flat, series):
        for label, rows in datasets:
            frames = sorted(rows)
            values = [
                numeric if (numeric := value(rows[frame], *keys)) is not None else np.nan
                for frame in frames
            ]
            axis.plot(frames, values, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize="small")

    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    fig.tight_layout()


def plot_kalman_comparisons(
    datasets: list[tuple[str, dict[int, dict[str, str]]]],
) -> None:
    """Wykres 2: Filtrowane dane i predykcje z Filtra Kalmana względem referencji (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("Kalman Filter State Estimation vs Reference", fontsize=14, fontweight="bold")
    
    # Mapujemy zarówno nazwy z Kalmana jak i odpowiadające im nazwy z referencji (ref_)
    series = [
        (("kalman_velocity_m_s", "ref_velocity_m_s"), "Velocity [m/s]"),
        (("kalman_x", "ref_x"), "X Position [m]"),
        (("kalman_y", "ref_y"), "Y Position [m]"),
        (("kalman_z", "ref_z"), "Z Position [m]"),
    ]

    for axis, (keys, ylabel) in zip(axes.flat, series):
        for label, rows in datasets:
            frames = sorted(rows)
            values = [
                numeric if (numeric := value(rows[frame], *keys)) is not None else np.nan
                for frame in frames
            ]
            axis.plot(frames, values, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize="small")

    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CFAR/MUSIC and Kalman filter outputs against reference data."
    )
    parser.add_argument(
        "result_files",
        nargs="+",
        type=Path,
        help="Two or more CSV files. The first is used as the baseline.",
    )
    args = parser.parse_args()

    if len(args.result_files) < 2:
        raise ValueError("Provide at least two CSV files.")
    datasets = []
    for path in args.result_files:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        datasets.append((display_label(path), normalize_rows(path)))

    base_label, base = datasets[0]
    print(f"Baza: {base_label}")
    for label, candidate in datasets[1:]:
        metrics = compare_to_base(base, candidate)
        print(f"\nPorównanie: {label} względem {base_label}")
        print(f"Wspólne klatki: {metrics['common_frames']}")
        for field in ("range_m", "velocity_m_s", "azimuth_deg", "elevation_deg", "kalman_velocity_m_s", "kalman_x", "kalman_y", "kalman_z"):
            if f"{field}_mae" in metrics and metrics[f'{field}_mae'] is not None:
                print(f"{field} MAE: {metrics[f'{field}_mae']:.4f}")
                print(f"{field} bias: {metrics[f'{field}_bias']:.4f}")
                
    # Generowanie obu zestawów wykresów w osobnych oknach
    plot_cfar_music_comparisons(datasets)
    plot_kalman_comparisons(datasets)
    
    plt.show()


if __name__ == "__main__":
    main()