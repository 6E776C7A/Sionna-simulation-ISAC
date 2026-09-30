import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial

def setup_scene(scene_file, fc, rx_rows, rx_cols, radar_position=[-9.169, 87.045, 17.500], target_position=[0.0, 0.0, 10.0]):
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

            if "dron" in name_lower:
                obj.radio_material = "drone_target_material"
                # print(f'Assigned drone material to object: {name}')
                
            elif "samochod_target" in name_lower:
                obj.radio_material = "car_target_material"
                # print(f'Assigned car target material to object: {name}')

            elif "czlowiek_target" in name_lower:
                obj.radio_material = "human_target_material"
                #print(f'Assigned human target material to object: {name}')

            elif "samochod" in name_lower:
                obj.radio_material = "car_material"
                # print(f'Assigned car material to object: {name}')

            elif "czlowiek" in name_lower:
                obj.radio_material = "human_material"
                # print(f'Assigned human material to object: {name}')

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

    obj.position += velocity * frame_time

    vx, vy, vz = velocity
    horizontal_speed = np.sqrt(vx**2 + vy**2)

    yaw = np.arctan2(vy, vx)

    pitch = np.arctan2(-vz, horizontal_speed) if horizontal_speed > 0 else 0.0

    obj.orientation = [float(yaw), float(pitch), 0.0]