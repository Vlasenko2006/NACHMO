import os
import numpy as np
import matplotlib.pyplot as plt
import re

def substitute_pattern(input_string, old_pattern, new_pattern):
    """
    This function takes an input string, searches for a specific pattern, and substitutes it with another pattern.
    If the pattern does not exist, it returns the initial string.

    :param input_string: The input string where the pattern will be searched and substituted.
    :param old_pattern: The pattern to search for in the input string.
    :param new_pattern: The pattern to substitute in place of the old pattern.
    :return: The modified string with the pattern substituted, or the initial string if the pattern does not exist.
    """
    if re.search(old_pattern, input_string):
        return re.sub(old_pattern, new_pattern, input_string)
    return input_string

def read_rmse_files(folder_path):
    # List to store the filenames and their corresponding values
    results = []
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file ends with '_rmse.npy'
        if filename.endswith('_rmse.npy'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Load the .npy file
            values = np.load(file_path)
            # Ensure the file contains exactly 3 values
            if len(values) == 3:
                # Remove 'OH_' from the start and '_rmse' from the end of the filename
                cleaned_filename = filename.replace('OH_', '').replace('_rmse.npy', '')
                # Append the cleaned filename and values to the results list
                cleaned_filename = substitute_pattern(cleaned_filename, "no_rec_no_gate", "$nrg$")
                cleaned_filename = substitute_pattern(cleaned_filename, "Sp", "$S_{p}$")
                cleaned_filename = substitute_pattern(cleaned_filename, "S_ST", "$S_{ST}$")
                cleaned_filename = substitute_pattern(cleaned_filename, "S_SVD", "$S_{SVD}$")
                cleaned_filename = substitute_pattern(cleaned_filename, "no_gate", "ng")
                cleaned_filename = substitute_pattern(cleaned_filename, "no_rec", "$nr$")
                cleaned_filename = substitute_pattern(cleaned_filename, "no_rec_no_gate", "$nrg$")
                results.append((cleaned_filename, values.tolist()))
            else:
                print(f"File {filename} does not contain exactly 3 values.")
    
    return results

def plot_rmse_data(rmse_data, slices=["1", "2", "3"], fig_name="noname.pdf"):
    # Extract the names and values
    names = [item[0] for item in rmse_data]
    values = [item[1] for item in rmse_data]
    
    # Define the bar width
    bar_width = 0.2
    
    # Sort the groups based on the third (blue) bar in descending order
    sorted_indices = np.argsort([v[2] for v in values])[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Set positions for the bars
    r1 = np.arange(len(sorted_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Plotting the bars
    plt.figure(figsize=(12, 6))
    
    for i, v in enumerate(sorted_values):
        plt.bar(r1[i], v[0], color='green', width=bar_width, edgecolor='grey', label=slices[0] if i == 0 else "")
        plt.bar(r2[i], v[1], color='red', width=bar_width, edgecolor='grey', label=slices[1] if i == 0 else "")
        plt.bar(r3[i], v[2], color='blue', width=bar_width, edgecolor='grey', label=slices[2] if i == 0 else "")
    
    # Adding X and Y axis labels
    plt.xlabel('NN\'s configuration', fontweight='bold')
    plt.ylabel('RMSE', fontweight='bold')
    plt.yscale('log')
    plt.xticks([r + bar_width for r in range(len(sorted_names))], sorted_names, rotation=90)
    
    # Adding the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Show the plot
    plt.tight_layout()  # Adjust layout to make room for the rotated x-ticks
    plt.savefig(fig_name + ".pdf")
    plt.show()




folder_with_data = 'SMSE3'
option = ["OH", "Verwer"]
training = ["_singlestep","_multistep"]
slices_oh = ["20 steps", "3000 steps", "6000 steps"]
slices_verwer = ["5 steps", "60 steps", "120 steps"]
top_dir_output = "barplots"
for opt in option:
    for tr in training:
        if opt == "OH":
            slices = slices_oh
        if opt == "Verwer":
            slices = slices_verwer
        folder_path = folder_with_data + opt + tr + "/"
        rmse_files_data = read_rmse_files(folder_path)
        if not os.path.exists(top_dir_output):
            os.makedirs(top_dir_output)
        file_path = os.path.join(top_dir_output, opt + tr)
        plot_rmse_data(rmse_files_data, slices, file_path)