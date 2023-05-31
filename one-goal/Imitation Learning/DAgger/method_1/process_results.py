import json
import numpy as np

for i in range(0, 36):
    file_name = f"/vol/bitbucket/av1019/dagger/hyperparameters/final_tuning_resultslts/final_tuning_hp_{i}.json"

    # Read JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)

########################### CONVERT TO DEGREES AND MILLIMETERS ###########################

    # Convert orientation error to degrees and distance error to millimeters
    for entry in data:
        entry['orientation_error'] = entry['orientation_error'] * 180 / np.pi  # Convert radians to degrees
        entry['distance_error'] = entry['distance_error'] * 1000  # Convert meters to millimeters

    # Calculate average distance error and orientation error
    avg_distance_error = sum(entry['distance_error'] for entry in data) / len(data)
    avg_orientation_error = sum(entry['orientation_error'] for entry in data) / len(data)

    # Add average errors to dictionary
    average_errors = {
        'average_distance_error': avg_distance_error,
        'average_orientation_error': avg_orientation_error
    }

    data = data + [average_errors]

########################### ROUND TO 1 DECIMAL PLACE ###########################

    # Round distance and orientation errors to 1 decimal place
    for entry in data[:-1]:  # Exclude the average errors entry
        entry['distance_error'] = round(entry['distance_error'], 1)
        entry['orientation_error'] = round(entry['orientation_error'], 1)

    # Round average errors to 1 decimal place
    data[-1]['average_distance_error'] = round(data[-1]['average_distance_error'], 1)
    data[-1]['average_orientation_error'] = round(data[-1]['average_orientation_error'], 1)

################################# FORMAT AS LATEX #################################

    # Extract distance and orientation errors for each item
    errors = []
    for entry in data[:-1]:
        errors.append(entry['distance_error'])
        errors.append(entry['orientation_error'])

    # Add average errors at the end
    errors.append(data[-1]['average_distance_error'])
    errors.append(data[-1]['average_orientation_error'])

    # Format errors as a string
    formatted_errors = ' & '.join([str(error) for error in errors])

    # Append formatted errors to the JSON data
    data.append({"latex": formatted_errors})

    # Write updated data with average errors to JSON file
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
