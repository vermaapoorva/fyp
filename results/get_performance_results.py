import json
import os

import numpy as np

def get_sac_data(x_range):

    base_directory = "final_results/sac_final_results/"    

    data = {}

    for amount_of_data in x_range:
        for scene in range(0, 5):
            for seed in [1019, 2603, 210423]:
                file_name = f"{base_directory}actual_final_results_{amount_of_data}_scene_{scene}_seed_{seed}.json"
                # skip if file doesn't exist
                if not os.path.isfile(file_name):
                    print("Skipping file: ", file_name)
                    continue
                
                print("Reading file: ", file_name)
                with open(file_name, 'r') as f:
                    results = json.load(f)
                    distance_errors = results[0]['distance_errors']
                    orientation_errors = results[0]['orientation_errors']

                    # convert distance errors from m to mm
                    distance_errors = [error * 1000 for error in distance_errors]
                    # convert orientation errors from rad to deg
                    orientation_errors = [error * 180 / np.pi for error in orientation_errors]

                    if amount_of_data not in data:
                        data[amount_of_data] = {scene: [[], []]}
                    elif scene not in data[amount_of_data]:
                        data[amount_of_data][scene] = [[], []]
                    
                    data[amount_of_data][scene][0].extend(distance_errors)
                    data[amount_of_data][scene][1].extend(orientation_errors)

            # save data to file
            with open(f"sac_final_data.json", 'w') as f:
                json.dump(data, f, indent=4)
    return data

def get_ppo_data(x_range):
    base_directory = "/vol/bitbucket/av1019/PPO/final_models/results/"    

    data = {}

    for amount_of_data in x_range:
        for scene in range(0, 5):
            for seed in [1019, 2603, 210423]:
                file_name = f"{base_directory}actual_final_results_{amount_of_data}_scene_{scene}_seed_{seed}.json"
                # skip if file doesn't exist
                if not os.path.isfile(file_name):
                    print("Skipping file: ", file_name)
                    continue
                
                print("Reading file: ", file_name)
                with open(file_name, 'r') as f:
                    results = json.load(f)
                    distance_errors = results[0]['distance_errors']
                    orientation_errors = results[0]['orientation_errors']

                    # convert distance errors from m to mm
                    distance_errors = [error * 1000 for error in distance_errors]
                    # convert orientation errors from rad to deg
                    orientation_errors = [error * 180 / np.pi for error in orientation_errors]

                    if amount_of_data not in data:
                        data[amount_of_data] = {scene: [[], []]}
                    elif scene not in data[amount_of_data]:
                        data[amount_of_data][scene] = [[], []]
                    
                    data[amount_of_data][scene][0].extend(distance_errors)
                    data[amount_of_data][scene][1].extend(orientation_errors)

            # save data to file
            with open(f"ppo_final_data.json", 'w') as f:
                json.dump(data, f, indent=4)
    return data

def get_bc_data():
    base_directory = "/vol/bitbucket/av1019/behavioural-cloning/c2f/final_results/"

    data = {}

    scene_names = ["cutlery_block_scene", "wooden_block_scene", "bowl_scene", "teapot_scene", "purple_block_scene"]
    amount_of_data_strings = ["10k", "100k", "1M", "5M", "10M"]
    for amount_of_data_index, amount_of_data in enumerate([10000, 100000, 1000000, 5000000, 10000000]):
        for scene in range(0, 4):
            file_name = f"{base_directory}final_model_{amount_of_data_strings[amount_of_data_index]}_fix_val_{scene_names[scene]}.json"
            # skip if file doesn't exist
            if not os.path.isfile(file_name):
                print("Skipping file: ", file_name)
                continue
            
            print("Reading file: ", file_name)
            with open(file_name, 'r') as f:
                results = json.load(f)
                distance_errors = results[0]['distance_errors']
                orientation_errors = results[0]['orientation_errors']

                # convert distance errors from m to mm
                distance_errors = [error * 1000 for error in distance_errors]
                # convert orientation errors from rad to deg
                orientation_errors = [error * 180 / np.pi for error in orientation_errors]

                if amount_of_data not in data:
                    data[amount_of_data] = {scene: [[], []]}
                elif scene not in data[amount_of_data]:
                    data[amount_of_data][scene] = [[], []]
                
                data[amount_of_data][scene][0].extend(distance_errors)
                data[amount_of_data][scene][1].extend(orientation_errors)

        # save data to file
        with open(f"bc_final_data.json", 'w') as f:
            json.dump(data, f, indent=4)
    return data

def calculate_mean_std(data, average_over_scenes):
    means_all_distances = []
    stds_all_distances = []
    means_all_orientations = []
    stds_all_orientations = []
    for amount_of_data, scene_errors in data.items():
        mean_distances = []
        mean_orientations = []
        std_distances = []
        std_orientations = []

        if average_over_scenes is not None:
            distances = np.array([])
            orientations = np.array([])
            for scene, errors in scene_errors.items():
                print(f"scene: {scene}")
                print(f"average_over_scenes: {average_over_scenes}")
                if int(scene) not in average_over_scenes:
                    print("Skipping scene: ", scene)
                    continue
                distances = np.append(distances, errors[0])
                orientations = np.append(orientations, errors[1])
            mean_distance = np.mean(distances)
            mean_orientation = np.mean(orientations)
            std_distance = np.std(distances)
            std_orientation = np.std(orientations)

            mean_distances.append(mean_distance)
            mean_orientations.append(mean_orientation)
            std_distances.append(std_distance)
            std_orientations.append(std_orientation)

        else:
            for scene, errors in scene_errors.items():
                print(f"scene: {scene}")
                distances = np.array(errors[0])
                orientations = np.array(errors[1])
                mean_distance = np.mean(distances)
                mean_orientation = np.mean(orientations)
                std_distance = np.std(distances)
                std_orientation = np.std(orientations)

                mean_distances.append(mean_distance)
                mean_orientations.append(mean_orientation)
                std_distances.append(std_distance)
                std_orientations.append(std_orientation)

        mean_distances = np.array(mean_distances)
        mean_orientations = np.array(mean_orientations)
        std_distances = np.array(std_distances)
        std_orientations = np.array(std_orientations)

        means_all_distances.append(mean_distances)
        means_all_orientations.append(mean_orientations)
        stds_all_distances.append(std_distances)
        stds_all_orientations.append(std_orientations)
    
    means_all_distances = np.array(means_all_distances)
    means_all_orientations = np.array(means_all_orientations)
    stds_all_distances = np.array(stds_all_distances)
    stds_all_orientations = np.array(stds_all_orientations)

    # Transpose
    means_all_distances = np.transpose(means_all_distances)
    means_all_orientations = np.transpose(means_all_orientations)
    stds_all_distances = np.transpose(stds_all_distances)
    stds_all_orientations = np.transpose(stds_all_orientations)

    return means_all_distances, means_all_orientations, stds_all_distances, stds_all_orientations

def generate_json_files(data, algo):
    for avg_over_scenes in [True, False]:
        for include_purple_block in [True, False]:

            if include_purple_block:
                include_purple_block_text = ''
                scenes = [0, 1, 2, 3, 4]
            else:
                include_purple_block_text = '_without_purple_block'
                scenes = [0, 1, 2, 3]
            
            if avg_over_scenes:
                avg_text = 'average_over_scenes'
            else:
                avg_text = 'per_scene'
                scenes = None
            
            file_name = f"data_to_plot/{algo}_data_to_graph_{avg_text}{include_purple_block_text}"

            # Extract the error statistics from the JSON files
            means_all_distances, means_all_orientations, stds_all_distances, stds_all_orientations = calculate_mean_std(data, scenes)

            # create one results and store in json file
            results = {}
            results['means_all_distances'] = means_all_distances.tolist()
            results['means_all_orientations'] = means_all_orientations.tolist()
            results['stds_all_distances'] = stds_all_distances.tolist()
            results['stds_all_orientations'] = stds_all_orientations.tolist()

            with open(f"{file_name}.json", 'w') as f:
                json.dump(results, f, indent=4)

def generate_npy_files(algo):

    for avg_over_scenes in [True, False]:
        for include_purple_block in [True, False]:

            if include_purple_block:
                include_purple_block_text = ''
                scenes = [0, 1, 2, 3, 4]
            else:
                include_purple_block_text = '_without_purple_block'
                scenes = [0, 1, 2, 3]
            
            if avg_over_scenes:
                avg_text = 'average_over_scenes'
            else:
                avg_text = 'per_scene'
                scenes = None
            
            file_name = f"data_to_plot/{algo}_data_to_graph_{avg_text}{include_purple_block_text}"

            # Extract the error statistics from the JSON files
            with open(f"{file_name}.json", 'r') as f:
                data = json.load(f)

            results = np.array([np.array(data['means_all_distances']), np.array(data['means_all_orientations']), np.array(data['stds_all_distances']), np.array(data['stds_all_orientations'])])
            np.save(f"{file_name}.npy", results)

algo = 'sac'
x_range = [500000*i for i in range(1, 21)]

data = get_sac_data(x_range)
generate_json_files(data, algo)