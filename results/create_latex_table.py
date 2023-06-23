import json

algo = 'sac'

# Read the JSON file
with open(f'/home/apoorva/Documents/FYP/results/data_to_plot/{algo}_data_to_graph_average_over_scenes_without_purple_block.json') as file:
    data = json.load(file)

# Extract the required data
means_all_distances = data['means_all_distances']
stds_all_distances = data['stds_all_distances']
means_all_orientations = data['means_all_orientations']
stds_all_orientations = data['stds_all_orientations']

# Round all values to 1 decimal place
means_all_distances = [[round(x, 1) for x in y] for y in means_all_distances]
stds_all_distances = [[round(x, 1) for x in y] for y in stds_all_distances]
means_all_orientations = [[round(x, 1) for x in y] for y in means_all_orientations]
stds_all_orientations = [[round(x, 1) for x in y] for y in stds_all_orientations]

# Create the LaTeX format
latex = ''

# 10k, 100k, 500k, 1M, 1.5M, ....
for j in [0, 1, 3]:
    print(f"amount of data: {j}")
    latex += f"{means_all_distances[0][j]} $\\pm$ {stds_all_distances[0][j]} & "
    latex += f"{means_all_orientations[0][j]} $\\pm$ {stds_all_orientations[0][j]} & "

# Remove the last '&' character
latex = latex.rstrip(' & ')

# Print the final LaTeX format
print(latex)
