import csv

import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial

def setup_scene(scene_file, fc, rx_rows, rx_cols, radar_position=[-9.169, 87.045, 17.500], target_position=[0.0, 0.0, 10.0]):
    """
    Set up the scene with the specified parameters.

    Parameters:
    - scene_file: Path to the scene file.
    - fc: Carrier frequency.
    - rx_rows: Number of rows in the receiver array.
    - rx_cols: Number of columns in the receiver array.
    - radar_position: Position of the radar (default: [-9.169, 87.045, 17.500]).
    - target_position: Position of the target (default: [0.0, 0.0, 10.0]).
    
    Returns:
    - scene: The loaded scene object.
    - tx: The transmitter object.
    - rx: The receiver object.
    """
    # Wczytanie sceny
    scene = load_scene(scene_file, merge_shapes=False)
    scene.frequency = fc
    
    # Tworzenie materiałów radiowych i przypisanie ich do obiektów w scenie
    human_material = RadioMaterial("human_material", relative_permittivity=16.0, conductivity=25.0, scattering_coefficient=0.5, color=[1.0, 0.4, 0.8])
    car_material = RadioMaterial("car_material", relative_permittivity=1.0, conductivity=1e7, scattering_coefficient=0.3, color=[0.5, 0.5, 0.5]) 
    drone_target_material = RadioMaterial("drone_target_material", relative_permittivity=1.0, conductivity=1e7, scattering_coefficient=0.4, color=[178/255.0, 34/255.0, 34/255.0])
    car_target_material = RadioMaterial("car_target_material", relative_permittivity=1.0, conductivity=1e7, scattering_coefficient=0.3, color=[178/255.0, 34/255.0, 34/255.0])
    human_target_material = RadioMaterial("human_target_material", relative_permittivity=16.0, conductivity=25.0, scattering_coefficient=0.5, color=[178/255.0, 34/255.0, 34/255.0])

    for mat in [human_material, car_material, drone_target_material, car_target_material, human_target_material]:
            if mat.name not in scene.radio_materials:
                scene.add(mat)

    for name, obj in scene.objects.items():
            name_lower = name.lower()


            if "samochod_target_d" in name_lower:
                obj.radio_material = "car_target_material"
                # print(f'Assigned car target material to object: {name}')
            
            elif "samochod_target_b" in name_lower:
                obj.radio_material = "car_target_material"
                # print(f'Assigned car target material to object: {name}')
            
            elif "dron" in name_lower:
                obj.radio_material = "drone_target_material"
                # print(f'Assigned drone material to object: {name}')

            elif "czlowiek_target" in name_lower:
                obj.radio_material = "human_target_material"
                #print(f'Assigned human target material to object: {name}')

            elif "samochod" in name_lower:
                obj.radio_material = "car_material"
                # print(f'Assigned car material to object: {name}')

            elif "czlowiek" in name_lower:
                obj.radio_material = "human_material"
                # print(f'Assigned human material to object: {name}')

    # Nadpisanie współczynnika rozproszenia dla wbudowanych materiałów ITU
    if "itu_marble" in scene.radio_materials:
        scene.radio_materials["itu_marble"].scattering_coefficient = 0.3
    
    if "itu_metal" in scene.radio_materials:
        scene.radio_materials["itu_metal"].scattering_coefficient = 0.1

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=rx_rows, num_cols=rx_cols, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")

    tx_pos = np.array(radar_position)
    target = np.array(target_position)
    direction = target - tx_pos

    yaw = np.arctan2(direction[1], direction[0])
    distance = np.linalg.norm(direction[:2])
    pitch = np.arctan2(-direction[2], distance)

    tx = Transmitter(name="tx", position=radar_position, orientation=[float(yaw), float(pitch), 0.0], display_radius=2.0)
    rx = Receiver(name="rx", position=radar_position,orientation=[float(yaw), float(pitch), 0.0], display_radius=2.0)

    scene.add(tx)
    scene.add(rx)

    return scene, tx, rx 

def update_object_position(obj, velocity, frame_time):

    obj.velocity = velocity
    obj.position = obj.position + velocity * frame_time

    vx, vy, vz = velocity
    horizontal_speed = np.sqrt(vx**2 + vy**2)

    if horizontal_speed > 0 or vz != 0:
        yaw = np.arctan2(vy, vx) if horizontal_speed > 0 else obj.orientation[0]
        pitch = np.arctan2(-vz, horizontal_speed) if horizontal_speed > 0 else 0.0
        obj.orientation = [float(yaw), float(pitch), 0.0]
        
def save_reference_data(target_object, rx, frame_id, time_s, target_id):
    """
    Save reference data for a target object.

    Parameters:
    - target_object: The target object for which to save reference data.
    - rx: The receiver object.
    - frame_id: The current frame ID.
    - time_s: The current time in seconds.
    - target_id: The ID of the target object.

    Returns:
    reference_data: A dictionary containing the reference data for the target object.
    """
    target_position = np.asarray(target_object.position, dtype=float).reshape(-1)
    target_velocity = np.asarray(target_object.velocity, dtype=float).reshape(-1)
    radar_position = np.asarray(rx.position, dtype=float).reshape(-1)

    if target_position.size != 3 or target_velocity.size != 3 or radar_position.size != 3:
        raise ValueError(
            "Target position, target velocity, and radar position must each contain 3 values."
        )

    relative_position = target_position - radar_position

    yaw, pitch, _ = np.asarray(rx.orientation, dtype=float).reshape(-1)

    rotation_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    rotation_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    rotation_global_to_radar = rotation_z @ rotation_y

    relative_position = rotation_global_to_radar.T @ relative_position

    range_m = np.linalg.norm(relative_position)

    if range_m == 0:
         raise ValueError("Target and radar positions are the same, resulting in zero range.")

    line_of_sight = relative_position / range_m

    target_velocity = rotation_global_to_radar.T @ target_velocity

    radial_velocity = np.dot(target_velocity, line_of_sight)

    dx, dy, dz = relative_position
    azimuth_deg = np.degrees(np.arctan2(dy, dx))
    elevation_deg = np.degrees(np.arctan2(dz, np.hypot(dx, dy)))

    reference_data = {
        "frame_id": int(frame_id),
        "time_s": round(float(time_s), 1),
        "target_id": target_id,

        "ref_x": round(float(dx), 6),
        "ref_y": round(float(dy), 6),
        "ref_z": round(float(dz), 6),

        "ref_vx": round(float(target_velocity[0]), 6),
        "ref_vy": round(float(target_velocity[1]), 6),
        "ref_vz": round(float(target_velocity[2]), 6),

        "ref_velocity_m_s": round(float(np.linalg.norm(target_velocity)), 6),

        "range_m": round(float(range_m), 6),
        "radial_velocity_m_s": round(float(radial_velocity), 6),
        "azimuth_local_deg": round(float(azimuth_deg), 6),
        "elevation_local_deg": round(float(elevation_deg), 6)
    }

    return reference_data

def save_reference_data_to_file(reference_data_list, filename):
    """
    Save a list of reference data dictionaries to a CSV file.

    Parameters:
    - reference_data_list: A list of dictionaries containing reference data.
    - filename: The name of the CSV file to save the data to.

    Returns:
    None
    """

    if not reference_data_list:
        raise ValueError("The reference data list is empty.")

    fieldnames = list(reference_data_list[0].keys())

    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reference_data_list:
            formatted_row = {}

            for key, value in row.items():
                if key == "time_s":
                    formatted_row[key] = f"{float(value):.1f}"
                elif isinstance(value, float):
                    formatted_row[key] = f"{value:.6f}"
                else:
                    formatted_row[key] = value

            writer.writerow(formatted_row)