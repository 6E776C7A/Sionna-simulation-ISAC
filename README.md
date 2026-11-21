# Sionna-simulation-mmWave

High-fidelity 28 GHz mmWave Integrated Sensing and Communications (ISAC) radar simulation and signal processing pipeline built with **Sionna RT**. The project evaluates target detection, localization, and tracking in a realistic urban environment (AGH University campus, Kraków).

## Key Features

* **Urban Ray-Traced Channel Modeling:** 28 GHz OFDM Channel Frequency Response (CFR) generation using Sionna RT with specular and diffuse reflections (`max_depth=3`, `1_000_000` rays/frame).
* **Dynamic Target Scenarios:** Individual and multi-target trajectories for vehicles (`car_B`, `car_D`), UAV (`drone`), and pedestrians (`human`) with velocity-aware Doppler modeling.
* **Ground-Truth & LoS/NLoS Labeling:** Automated export of reference trajectories (`.csv`) including Mitsuba ray-casting Line-of-Sight (`is_los`) verification per frame.
* **Reproducible & Memory-Efficient Storage:** Fixed random seeds for statistical repeatability and chunked CFR export to compressed `.npz` archives (`complex64`, 16 RX antennas × 252 OFDM symbols × 792 subcarriers).
* **Radar DSP Pipeline:** Complete post-processing chain covering MTI clutter suppression, Range-Doppler estimation, CFAR target detection, MUSIC angle-of-arrival estimation, Kalman filter tracking, and RMSE evaluation.

## Repository Structure

```text
├── simulation/            # Sionna RT notebooks & scene configuration
├── dsp/                   # Radar DSP modules (MTI, RD, CFAR, MUSIC, Kalman)
├── data/                  # Local datasets, reference CSVs & results (Git-ignored)
├── ofdm_config.py         # Shared OFDM & radar parameters
├── main.py                # Full multi-frame processing pipeline
└── main_single_frame.py   # Single-frame inspection & debugging
```

## Workflow

### 1. Environment Setup
Install dependencies for local post-processing:
```bash
pip install -r requirements.txt
```
For running ray-tracing simulations (recommended on an NVIDIA GPU instance, e.g., RTX 4090), use `simulation/requirements.txt`.

### 2. Data Generation (Sionna RT)
Run the desired scenario notebook inside `simulation/`. Each simulation steps through 200 frames (`frame_time = 0.1 s`), computes chunked CFR tensors, and exports:
* `simulation_data_<scenario>_<part>.npz` — compressed `complex64` CFR batches.
* `ref_<scenario>.csv` — ground-truth positions, radial velocities, and `is_los` flags.

### 3. Radar Post-Processing
Place the generated `.npz` and `.csv` files in your local data directory and run:
```bash
python main.py
```
For inspecting individual frames (Range-Doppler maps, CFAR detections, MUSIC spectrum), use `main_single_frame.py`.