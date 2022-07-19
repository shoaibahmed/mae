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
    cm = plt.get_cmap('plasma')
    marker_colors = [cm(1.*i/len(model_names)) for i in range(len(model_names))]
    # marker_colors = ['green', 'red', 'orange', 'purple', 'magenta', 'blue']
    # marker_colors = ['royalblue', 'darkblue']
    marker_colors_second = ['limegreen', 'darkgreen']

    # Generate bars with stride
    bar_width = 0.8 / len(model_names)
    start_point = - bar_width * (len(model_names) / 2) + 0.5 * bar_width
    
    # Plot the results
    fig, ax = plt.subplots()
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
    
    plot_width = 6 * len(model_names)
    if len(model_names) == 1:
        plot_width = 12
    fig.set_size_inches(plot_width, 8)
    
    r_list = []
    for i in range(len(model_names)):
        stride = start_point + i*bar_width
        r = [x + stride for x in np.arange(len(corruption_types))]
        r_list.append(r)

    num_expected_ignore = 2  # train and validation stats
    marker_select =0
    for i, (key, results) in enumerate(all_results.items()):
        error_list = []
        error_std_list = []
        accuracy_list = []
        accuracy_std_list = []
        use_idx = []
        is_acc = "acc" in key.lower()
        for j, corruption_type in enumerate(corruption_types):
            corruption_levels = list(results[corruption_type].keys())
            assert all([x == str(y) for x, y in zip(corruption_levels, range(1, 6))])
            acc_list = [results[corruption_type][corruption_severity] for corruption_severity in corruption_levels]
            acc_list = [x if x is not None else 0. for x in acc_list]  # Filter
            # acc_list = [results[corruption_type][corruption_severity][0] for corruption_severity in corruption_levels]
            # loss_list = [results[corruption_type][corruption_severity][1] for corruption_severity in corruption_levels]
            # cls_loss_list = [results[corruption_type][corruption_severity][2] for corruption_severity in corruption_levels]
            # recons_loss_list = [results[corruption_type][corruption_severity][3] for corruption_severity in corruption_levels]

            # Filter for none vals
            # cls_loss_list = [x if x is not None else 0. for x in cls_loss_list]
            # recons_loss_list = [x if x is not None else 0. for x in recons_loss_list]

            if is_acc:
                error_rate = [100 - x for x in acc_list]
            else:
                error_rate = acc_list
            mean_acc = np.mean(acc_list)
            print(f"Corruption type: {corruption_type} \t Mean accuracy: {mean_acc:.2f}%")
            
            if corruption_type.lower() in ["train", "validation"]:
                print("Ignoring class:", corruption_type)
            else:
                use_idx.append(j)
            
            accuracy_list.append(mean_acc)
            accuracy_std_list.append(np.std(acc_list))
            
            error_list.append(np.mean(error_rate))
            error_std_list.append(np.std(error_rate))
        
        # Draw the bars here
        alpha = 0.9
        if is_acc:
            ax.bar(r_list[i], error_list, yerr=error_std_list, width=bar_width, linewidth=2., color=marker_colors_second[marker_select % len(marker_colors_second)], alpha=alpha, label=key)
            marker_select += 1
        else:
            ax2.bar(r_list[i], error_list, yerr=error_std_list, width=bar_width, linewidth=2., color=marker_colors[i], alpha=alpha, label=key)
        
        assert len(corruption_types) - len(use_idx) == num_expected_ignore, len(corruption_types) - len(use_idx)
        overall_mean_error = np.mean([error_list[j] for j in use_idx])
        overall_mean_acc = np.mean([accuracy_list[j] for j in use_idx])
        print(f"Model file: {key} \t Mean stats \t Error: {overall_mean_error:.2f}% \t Accuracy: {overall_mean_acc:.2f}%")
    
    fontsize = 16
    x_pos = np.arange(len(corruption_types))
    plt.xticks(x_pos, rotation=90, fontsize=fontsize)
    ax.set_xticklabels(corruption_types)
    plt.yticks(fontsize=fontsize)
    plt.margins(x=0.02)

    ax.legend(loc='upper left', fontsize=fontsize)  # Acc legend
    ax2.legend(loc='upper right', fontsize=fontsize)  # Loss legend
    ax.set_ylabel("Error rate (%)", fontsize=fontsize)
    ax2.set_ylabel("Loss", fontsize=fontsize)
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
        
        acc = r["acc1"]
        loss = r["loss"]
        cls_loss, recons_loss = None, None
        if "cls_loss" in r:
            cls_loss = r["cls_loss"]
            recons_loss = r["recons_loss"]
        if isinstance(acc, dict):
            acc = acc["global_avg"]
            loss = loss["global_avg"]
            if cls_loss is not None:
                assert recons_loss is not None
                cls_loss = cls_loss["global_avg"]
                recons_loss = recons_loss["global_avg"]
        
        if is_noise_class:
            severity = r["severity"]
            results_dict[class_name][severity] = (acc, loss, cls_loss, recons_loss)
        else:
            results_dict[class_name] = {str(i): (acc, loss, cls_loss, recons_loss) for i in range(1, 6)}
    
    # Split the model name w.r.t. the title
    k1_sample = list(results_dict.keys())[0]
    k2_sample = list(results_dict[k1_sample].keys())[0]
    for i, loss_type in enumerate([" (Acc)", " (Loss)", " (CLS loss)", " (Recons loss)"]):
        if results_dict[k1_sample][k2_sample][i] is None:  # Don't add if empty
            print("Ignoring combination:", f"{title}{loss_type}")
            continue
        all_results[f"{title}{loss_type}"] = {}
        for k1 in results_dict:
            all_results[f"{title}{loss_type}"][k1] = {}
            for k2 in results_dict[k1]:
                all_results[f"{title}{loss_type}"][k1][k2] = results_dict[k1][k2][i]

print(all_results)
plot_results(all_results, output_file)
