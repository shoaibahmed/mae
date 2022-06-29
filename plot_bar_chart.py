import os
import sys
import natsort
import simplejson
import numpy as np
import matplotlib.pyplot as plt


def plot_results(all_results, file_name):
    # Number of models
    model_names = natsort.natsorted(list(all_results.keys()))
    print("Model names:", model_names)
    
    corruption_types = list(all_results[model_names[0]].keys())
    print("Corruption types:", corruption_types)
    
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
    # cm = plt.get_cmap('hsv')
    # marker_colors = [cm(1.*i/len(model_names)) for i in range(len(model_names))]
    marker_colors = ['green', 'red', 'orange', 'purple', 'magenta', 'blue']

    # Generate bars with stride
    bar_width = 0.8 / len(model_names)
    start_point = - bar_width * (len(model_names) / 2) + 0.5 * bar_width
    
    # Plot the results
    fig, ax = plt.subplots()
    plot_width = 6 * len(model_names)
    fig.set_size_inches(plot_width, 8)
    
    r_list = []
    num_expected_ignore = 2  # train and validation stats
    for i in range(len(model_names)):
        stride = start_point + i*bar_width
        r = [x + stride for x in np.arange(len(corruption_types)-num_expected_ignore)]
        r_list.append(r)

    for i, (key, results) in enumerate(all_results.items()):
        error_list = []
        error_std_list = []
        accuracy_list = []
        accuracy_std_list = []
        ignored_cls = 0
        for corruption_type in corruption_types:
            corruption_levels = list(results[corruption_type].keys())
            assert all([x == str(y) for x, y in zip(corruption_levels, range(1, 6))])
            acc_list = [results[corruption_type][corruption_severity] for corruption_severity in corruption_levels]

            error_rate = [100 - x for x in acc_list]
            mean_acc = np.mean(acc_list)
            print(f"Corruption type: {corruption_type} \t Mean accuracy: {mean_acc:.2f}%")
            
            if corruption_type.lower() in ["train", "validation"]:
                print("Ignoring class:", corruption_type)
                ignored_cls += 1
                continue
            
            accuracy_list.append(mean_acc)
            accuracy_std_list.append(np.std(acc_list))
            
            error_list.append(np.mean(error_rate))
            error_std_list.append(np.std(error_rate))
        
        # Draw the bars here
        plt.bar(r_list[i], error_list, yerr=error_std_list, width=bar_width, linewidth=2., color=marker_colors[i], alpha=0.6, label=key)
        
        assert ignored_cls == num_expected_ignore, ignored_cls
        overall_mean_error = np.mean(error_list)
        overall_mean_acc = np.mean(accuracy_list)
        print(f"Model file: {key} \t Mean stats \t Error: {overall_mean_error:.2f}% \t Accuracy: {overall_mean_acc:.2f}%")
    
    fontsize = 16
    x_pos = np.arange(len(corruption_types))
    plt.xticks(x_pos, rotation=90, fontsize=fontsize)
    ax.set_xticklabels(corruption_types)
    plt.yticks(fontsize=fontsize)
    plt.margins(x=0.02)

    plt.legend(fontsize=fontsize)
    plt.ylabel("Error rate (%)", fontsize=fontsize)
    plt.xlabel("Corruption type", fontsize=fontsize)
    # plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()


if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <Log files> <Legend title - seperated by ;> <Output file name>")
    exit()

file_names = sys.argv[1:-2]
legend_titles = sys.argv[-2].split(";")
output_file = sys.argv[-1]
assert len(file_names) == len(legend_titles), f"File names: {file_names} / Titles: {legend_titles}"

all_results = {}
for title, file_name in zip(legend_titles, file_names):
    print("Loading file:", file_name)
    assert os.path.exists(file_name)
    
    with open(file_name, "r") as f:
        lines = f.readlines()
    lines = [simplejson.loads(l.strip()) for l in lines if l[0] == '{']

    results_dict = {}
    for r in lines:
        is_noise_class = "noise_class" in r
        class_name = r["noise_class"] if is_noise_class else r["set"]
        if class_name not in results_dict:
            results_dict[class_name] = {}
        if is_noise_class:
            severity = r["severity"]
            acc = r["acc1"]
            results_dict[class_name][severity] = acc
        else:
            acc = r["acc1"]
            results_dict[class_name] = {str(i): acc for i in range(1, 6)}
    
    # dir_name = file_name.split(os.sep)[-2]
    all_results[title] = results_dict

print(all_results)
plot_results(all_results, output_file)
