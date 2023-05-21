import pickle

# Read the data from the pickle file
with open('model_accuracies/hp_tuning_net_arch_results.pkl', 'rb') as file:
    data = pickle.load(file)

# Sort the data by hyperparameter_index and scene_index
sorted_data = sorted(data, key=lambda x: (x[0], x[1]))

# Save the sorted data to a new pickle file
with open('model_accuracies/hp_tuning_net_arch_results_sorted.pkl', 'wb') as file:
    pickle.dump(sorted_data, file)

# Calculate the average of mean_pos_error and mean_or_error for each hyperparameter_index
averaged_data = []
current_hp_index = None
count = 0
sum_pos_error = 0
sum_or_error = 0

for item in sorted_data:
    hp_index, _, _, pos_error, or_error = item

    if hp_index != current_hp_index:
        if current_hp_index is not None:
            averaged_data.append([current_hp_index, sum_pos_error / count, sum_or_error / count])

        current_hp_index = hp_index
        count = 0
        sum_pos_error = 0
        sum_or_error = 0

    count += 1
    sum_pos_error += pos_error
    sum_or_error += or_error

# Append the last averaged data entry
if current_hp_index is not None:
    averaged_data.append([current_hp_index, sum_pos_error / count, sum_or_error / count])

# Save the averaged data to a new pickle file
with open('model_accuracies/hp_tuning_net_arch_results_averaged.pkl', 'wb') as file:
    pickle.dump(averaged_data, file)
