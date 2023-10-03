import os
import wandb
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def extract_data_from_wandb(
    wandb_path,
    group_name,
    key_name,
    value_name,
    smooth=10,
    max_key=True,
    best_k=None,
    exclude_runs=[],
):

    api = wandb.Api()
    runs = api.runs(wandb_path, filters={"group": group_name})
    print(len(runs))

    all_keys = []
    all_values = []
    for run in runs:
        if run.name in exclude_runs:
            print(f"run {run.name} is excluded")
            continue
        if run.state == "finished":
            keys = []
            values = []
            for row in run.scan_history():
                # check if "x" is in a key of row
                if value_name in row and row[value_name] is not None:
                    # print(row[value_name], row[key_name])
                    keys.append(row[key_name])
                    values.append(row[value_name])
                else:
                    pass
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
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["font.size"] = 20
    plt.subplots_adjust(left=0.165, right=0.99, bottom=0.16, top=0.95)
    plt.tight_layout()

    plt.plot(keys, means, label=label, color=color)
    plt.locator_params(nbins=10, axis="x")
    plt.locator_params(nbins=10, axis="y")
    # plt.ylim(0, 1050)

    plt.grid(alpha=0.8)
    # ax.title(title)
    plt.fill_between(keys, means - half_stds, means + half_stds, alpha=0.15)
    # plt.legend(loc="lower right", prop={"size": 6}).get_frame().set_edgecolor("0.1")
    # plt.legend(loc="upper left", ncol=1)
    plt.legend(ncol=1)
    plt.xlabel(key_name)
    plt.ylabel(value_name)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def plot_redundant_actions(
    game, wandb_path, value_name, key_name, suffix="", smooth=10, with_duel=True
):
    if value_name == "Episodic/reward":
        value_name_plot = "Reward"
    if key_name == "General/timesteps_done":
        key_name_plot = "Timesteps"
    if with_duel:
        labels = ["DVQN", "DQN", "Duel DQN", "DVQN-R", "DQN-R", "Duel DQN-R"]
    else:
        labels = ["DVQN", "DQN", "DVQN-R", "DQN-R"]
    file_name = f"/{value_name_plot}_{game}#{suffix}.png"

    if game == "Boxing-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Boxing
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Riverraid-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]
    elif game == "Asterix-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Breakout-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",  # Breakout
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",  # Breakout
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Pong-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]
    if with_duel:
        group_names += [
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel"
        ]
    # group_names += [
    #     "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_nAx3#2",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|dqn_nAx3#2",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|duel_nAx3",
    # ]
    # group_names += [
    #     "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_EachActionx5",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|dqn_EachActionx5",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|duel_EachActionx5",
    # ]
    # group_names += [
    #     "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_nAction30",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|dqn_nAction30",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|duel_nAction30",
    # ]
    if game == "Riverraid-v5":
        group_names += [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_nA+10NOOP_2",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|dqn_nA+10NOOP",
        ]
    else:
        group_names += [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_nA+10NOOP",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|dqn_nA+10NOOP",
        ]
    if with_duel:
        group_names += [
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|duel_nA+10NOOP",
        ]

    for i, group_name in enumerate(group_names):
        keys, means, half_stds = extract_data_from_wandb(
            wandb_path,
            group_name,
            key_name=key_name,
            value_name=value_name,
            smooth=smooth,
            max_key=False,
            best_k=None,
            exclude_runs=["resilient-deluge-1925", "sandy-glade-1920"],
        )

        plot_data(
            keys=keys,
            means=means,
            half_stds=half_stds,
            label=labels[i],
            # color=colors[i],
            key_name=key_name_plot,
            value_name=value_name_plot,
        )

    # plt.show()
    # if value_name_plot == "max Q(s,a)":
    #     plt.axhline(y=-1.0292871913294073, color="dimgrey", linestyle="dashed", linewidth=2.0)
    prefix = "/workspace/repos_dev/VQVAE_RL/plots/"
    path = os.path.join(prefix, f"{game}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + file_name)
    print("finished")

    plt.close()


def plot_TC(game, wandb_path, value_name, key_name, smooth=10):
    if value_name == "Episodic/reward":
        value_name_plot = "Reward"
    if key_name == "General/timesteps_done":
        key_name_plot = "Timesteps"
    labels = ["DVQN", "DQN", "DVQN+TC", "DQN+TC"]
    file_name = f"/{value_name_plot}_{game}#with_TC_test2.png"

    # group_names = [
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Breakout
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
    #     "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
    #     # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
    # ]
    if game == "Boxing-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Boxing
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Riverraid-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|#_*",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]
    elif game == "Asterix-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Breakout-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",  # Breakout
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",  # Breakout
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dqn+tc",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|#_*",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Pong-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|&dvqn_Vcur_10w",  # temp
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

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
            # color=colors[i],
            key_name=key_name_plot,
            value_name=value_name_plot,
        )

    # plt.show()
    # if value_name_plot == "max Q(s,a)":
    #     plt.axhline(y=-1.0292871913294073, color="dimgrey", linestyle="dashed", linewidth=2.0)
    prefix = "/workspace/repos_dev/VQVAE_RL/plots/"
    path = os.path.join(prefix, f"{game}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + file_name)
    print("finished")

    plt.close()


def plot_withCurl(game, wandb_path, value_name, key_name, smooth=10):
    if value_name == "Episodic/reward":
        value_name_plot = "Reward"
    if key_name == "General/timesteps_done":
        key_name_plot = "Timesteps"

    labels = ["DVQN", "DQN", "DVQN+Curl", "DQN+Curl"]
    file_name = f"/{value_name_plot}_{game}#with_Curl.png"

    # group_names = [
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Breakout
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
    #     "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
    #     # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
    # ]
    if game == "Boxing-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Boxing
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,raw,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_curl",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,raw,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Riverraid-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,raw,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_curl",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,raw,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]
    elif game == "Asterix-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,raw,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_curl",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,raw,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Breakout-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",  # Breakout
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",  # Breakout
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,raw,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_curl",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|#_*",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,raw,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Pong-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,raw,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dvqn_curl",  # temp
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,raw,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

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
            # color=colors[i],
            key_name=key_name_plot,
            value_name=value_name_plot,
        )

    # plt.show()
    # if value_name_plot == "max Q(s,a)":
    #     plt.axhline(y=-1.0292871913294073, color="dimgrey", linestyle="dashed", linewidth=2.0)
    prefix = "/workspace/repos_dev/VQVAE_RL/plots/"
    path = os.path.join(prefix, f"{game}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + file_name)
    print("finished")

    plt.close()


def plot_metric(
    game,
    wandb_path,
    value_name,
    key_name,
    file_name_suffix="",
    smooth=10,
):
    if value_name == "Episodic/reward":
        value_name_plot = "Reward"
    if value_name == "Info/grdQ/grd_q_max":
        value_name_plot = "max Q(s,a)"
    if key_name == "General/timesteps_done":
        key_name_plot = "Timesteps"

    labels = [r"DVQN $\alpha=0.5$", r"DVQN $\alpha=1.0$", "DQN", "DDQN", "CDDQN", "Duel DQN"]

    file_name = f"/{value_name_plot}_{game}#{file_name_suffix}.png"

    # group_names = [
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Breakout
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
    #     "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
    #     "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
    #     "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
    #     # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
    #     # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
    # ]
    if game == "Boxing-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w",  # Boxing
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|?dvqn_Vcur_10w",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|dvqn_alpha1.0",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Riverraid-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur",  # Riverraid
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|?dvqn_Vcur_10w",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|dvqn_alpha1.0",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|#_*",
            # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]
    elif game == "Asterix-v5":
        group_names = [
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2", #Boxing, Asterix
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|?dvqn_Vcur_10w",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|dvqn_alpha1.0",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Breakout-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",  # Breakout
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",  # Breakout
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|?dvqn_Vcur_10w",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|dvqn_alpha1.0",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@ddqn",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|@cddqn",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|dqn+tc",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|#_*",
            # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

    elif game == "Pong-v5":
        group_names = [
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|?dvqn_Vcur_10w#2",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|?dvqn_Vcur_10w",
            "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close1.0|dvqn_alpha1.0",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,raw,P0|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|@ddqn",
            "A0_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|@cddqn2",
            "A1_AEncD0_GEncD0_ShrEnc0_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.0|%duel",
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|&dvqn_Vcur_10w",  # temp
            # "A1_AEncD0_GEncD0_ShrEnc1_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.5|!",
            # "A1_AEncD0_GEncD0_ShrEnc0_Curl|grd,temp,P1|_VQ0|300,1.0,0,[0.0, 0.1, 0.0]|_bs128_ms100k_close0.0|!",
        ]

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
            # color=colors[i],
            key_name=key_name_plot,
            value_name=value_name_plot,
        )

    # plt.show()
    # if value_name_plot == "max Q(s,a)":
    #     plt.axhline(y=-1.0292871913294073, color="dimgrey", linestyle="dashed", linewidth=2.0)
    prefix = "/workspace/repos_dev/VQVAE_RL/plots/"
    path = os.path.join(prefix, f"{game}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + file_name)
    print("finished")

    plt.close()


if __name__ == "__main__":
    # value_name = "Info/grdQ/grd_q_max"
    # value_name_plot = "max Q(s,a)"
    value_name = "Episodic/reward"
    value_name_plot = "Reward"

    key_name = "General/timesteps_done"
    key_name_plot = "Timesteps"

    games = ["Breakout-v5", "Pong-v5", "Asterix-v5", "Boxing-v5", "Riverraid-v5"]
    # games = ["Asterix-v5"]

    for game in games:
        smooth = 10
        wandb_path = f"team-yuan/HDQN_Atari_{game}"
        # suffix = "redundant_EachActionX5", "redundant_nAction30, redundant_nA+10NOOP"
        # plot_redundant_actions(
        #     game,
        #     wandb_path,
        #     value_name,
        #     key_name,
        #     suffix="redundant_nA+10NOOP_2*",
        #     smooth=smooth,
        #     with_duel=True,
        # )

        # plot_withCurl(game, wandb_path, value_name, key_name, smooth=smooth)
        plot_metric(
            game,
            wandb_path,
            value_name,
            key_name,
            file_name_suffix="reward_rebuttal",
            smooth=smooth,
        )
