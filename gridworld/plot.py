import json
from matplotlib import pyplot as plt
import numpy as np
import scipy


def extract_data_logfile(
    log_path,
    key_name,
    value_name,
    smooth=10,
    max_key=True,
    interpolate=True,
):

    all_keys = []
    all_values = []
    keys = []
    values = []

    last_idice = []
    with open(log_path, "r") as f:
        # get the last line
        for line in f:
            pass
        last_line = line
        n_repeat = json.loads(line)["repeat_idx"] + 1

    with open(log_path, "r") as f:
        last_repeat_idx = 0
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            if data["repeat_idx"] != last_repeat_idx:
                last_idice.append(line_idx-1)
                last_repeat_idx = data["repeat_idx"]

        if line_idx != last_idice[-1]:
            last_idice.append(line_idx)

    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if key_name in data and value_name in data:
                keys.append(data[key_name])
                values.append(data[value_name])

            if i in last_idice:
                keys, values = np.array(keys), np.array(values)
                assert smooth < keys.shape[0]
                if smooth > 1 and values.shape[0] > 0:
                    K = np.ones(smooth)
                    ones = np.ones(values.shape[0])
                    values = np.convolve(values, K, "same") / np.convolve(ones, K, "same")
                all_keys.append(keys)
                all_values.append(values)
                keys = []
                values = []

    
    if interpolate:
        all_keys_tmp = sorted(all_keys, key=lambda x: x[-1])
        keys = all_keys_tmp[-1] if max_key else all_keys_tmp[0]
        # threshold = keys.shape[0]

        # interpolate
        for idx, (key, value) in enumerate(zip(all_keys, all_values)):
            f = scipy.interpolate.interp1d(key, value, fill_value="extrapolate")
            all_keys[idx] = keys
            all_values[idx] = f(keys)
    else:
        keys = all_keys[-1] 

    all_values = np.array(all_values)
    means = np.mean(all_values, axis=0)
    half_stds = 0.5 * np.std(all_values, axis=0)

    # means, half_stds = [], []
    # for i in range(threshold):
    #     vals = []

    #     for v in all_values:
    #         if i < v.shape[0]:
    #             vals.append(v[i])
    #     if best_k is not None:
    #         vals = sorted(vals)[-best_k:]
    #     means.append(np.mean(vals))
    #     # half_stds.append(0.5 * np.std(vals))
    #     half_stds.append(np.std(vals))

    # means = np.array(means)
    # half_stds = np.array(half_stds)

    # keys = all_keys[-1][:threshold]
    assert means.shape[0] == keys.shape[0]

    return keys, means, half_stds

def plot_data(
    keys,
    means,
    half_stds,
    max_time=None,
    label="DVQN",
    color=None,
    x_label=None,
    y_label=None,
    save_path=False,
):
    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    plt.rcParams["figure.figsize"] = (5, 3)
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    plt.tight_layout()

    plt.plot(keys, means, label=label, color=color)
    plt.locator_params(nbins=10, axis="x")
    plt.locator_params(nbins=10, axis="y")

    # make the largest y tick equal to 1
    # plt.yticks(np.linspace(-15, 1, 6))

    plt.grid(alpha=0.8)
    # ax.title(title)
    plt.fill_between(keys, means - half_stds, means + half_stds, alpha=0.15)
    # plt.legend(loc="lower right", prop={"size": 6}).get_frame().set_edgecolor("0.1")
    # plt.legend(loc="upper left", ncol=1)
    plt.legend(ncol=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # plt.ylim(top=5.0)
    # plt.ylim(-30, 5.0)
    if save_path:
        plt.savefig(f"{save_path}.png")

def plot_metric(log_path, key_name, value_name, x_label, y_label, label_name, interpolate=True, smooth=100, max_key=True, save_path=False):
    keys, means, half_stds = extract_data_logfile(    
        log_path,
        key_name,
        value_name,
        smooth=smooth,
        max_key=max_key,
        interpolate=interpolate,
        )
    pass
    plot_data(
        keys,
        means,
        half_stds,
        max_time=None,
        label=label_name,
        color=None,
        x_label=x_label,
        y_label=y_label,
        save_path=save_path,
    )