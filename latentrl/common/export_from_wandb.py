import os
import wandb
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def extract_data_from_wandb(
    wandb_path, group_name, key_name, value_name, smooth=10, max_key=True, best_k=None
):

    api = wandb.Api()
    runs = api.runs(wandb_path, filters={"group": group_name})
    print(len(runs))

    all_keys = []
    all_values = []
    for run in runs:
        if run.state == "finished":
            keys = []
            values = []
            for row in run.scan_history():
                # check if "x" is in a key of row
                if value_name in row:
                    # print(row[value_name], row[key_name])
                    keys.append(row[key_name])
                    values.append(row[value_name])
            keys, values = np.array(keys), np.array(values)
            if smooth > 1 and values.shape[0] > 0:
                K = np.ones(smooth)
                ones = np.ones(values.shape[0])
                values = np.convolve(values, K, "same") / np.convolve(ones, K, "same")
            all_keys.append(np.array(keys))
            all_values.append(np.array(values))

            # try:
            #     print(row["Info/grdQ/grd_q"])
            #     print(row["General/timesteps_done"])
            #     x.append(row["General/timesteps_done"])
            #     y.append(row["Info/grdQ/grd_q"])
            # except:
            #     pass
            # pass
            # break

    all_keys_tmp = sorted(all_keys, key=lambda x: x[-1])
    keys = all_keys_tmp[-1] if max_key else all_keys_tmp[0]
    threshold = keys.shape[0]

    # interpolate
    for idx, (key, value) in enumerate(zip(all_keys, all_values)):
        f = scipy.interpolate.interp1d(key, value, fill_value="extrapolate")
        all_keys[idx] = keys
        all_values[idx] = f(keys)

    means, half_stds = [], []
    for i in range(threshold):
        vals = []

        for v in all_values:
            if i < v.shape[0]:
                vals.append(v[i])
        if best_k is not None:
            vals = sorted(vals)[-best_k:]
        means.append(np.mean(vals))
        # half_stds.append(0.5 * np.std(vals))
        half_stds.append(np.std(vals))

    means = np.array(means)
    half_stds = np.array(half_stds)

    keys = all_keys[-1][:threshold]
    assert means.shape[0] == keys.shape[0]

    return keys, means, half_stds


def plot_data(
    keys,
    means,
    half_stds,
    max_time=None,
    label="DVQN",
    color=None,
    key_name=None,
    value_name=None,
):
    if max_time is not None:
        idxs = np.where(keys <= max_time)
        keys = keys[idxs]
        means = means[idxs]
        half_stds = half_stds[idxs]

    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 10
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    plt.tight_layout()

    plt.plot(keys, means, label=label, color=color)
    plt.locator_params(nbins=10, axis="x")
    plt.locator_params(nbins=10, axis="y")
    # plt.ylim(0, 1050)

    plt.grid(alpha=0.8)
    # ax.title(title)
    plt.fill_between(keys, means - half_stds, means + half_stds, color=color, alpha=0.15)
    # plt.legend(loc="lower right", prop={"size": 6}).get_frame().set_edgecolor("0.1")
    plt.legend(loc="upper center", ncol=4)
    plt.xlabel(key_name)
    plt.ylabel(value_name)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


if __name__ == "__main__":
    game = "Pong-v5"
    smooth = 10
    wandb_path = f"team-yuan/HDQN_Atari_{game}"
    group_names = [
        # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",
        "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
        "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|@ddqn",
        "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|@cddqn2",
    ]
    labels = ["DVQN", "DQN", "DDQN", "CDDQN"]
    colors = ["green", "black", "blue", "red"]

    # value_name = "Info/grdQ/grd_q_max"
    # value_name_plot = "max Q(s,a)"
    value_name = "Episodic/reward"
    value_name_plot = "Reward"

    key_name = "General/timesteps_done"
    key_name_plot = "Timesteps"

    for i, group_name in enumerate(group_names):
        keys, means, half_stds = extract_data_from_wandb(
            wandb_path,
            group_name,
            key_name=key_name,
            value_name=value_name,
            smooth=smooth,
            max_key=False,
            best_k=None,
        )

        plot_data(
            keys=keys,
            means=means,
            half_stds=half_stds,
            label=labels[i],
            color=colors[i],
            key_name=key_name_plot,
            value_name=value_name_plot,
        )

    # plt.show()
    if value_name_plot == "max Q(s,a)":
        plt.axhline(y=-1.0292871913294073, color="dimgrey", linestyle="dashed", linewidth=2.0)
    prefix = "/workspace/repos_dev/VQVAE_RL/plots/"
    path = os.path.join(prefix, f"{game}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + f"/{value_name_plot}_smooth{smooth}.png")
