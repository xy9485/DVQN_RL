import wandb

api = wandb.Api()

group_name = "A1_AEncD0_GEncD0_ShrEnc1_Curl|off,temp,P0|_VQ0|16,0.5,0,[0.0, 0.0, 0.0]|_bs128_ms100k_close0.5|^dvqn_Vcur"
runs = api.runs("team-yuan/HDQN_Atari_Pong-v5", filters={"group": group_name})
print(len(runs))
x = []
y = []
for run in runs:
    if run.state == "finished":
        for row in run.scan_history():
            # check if "x" is in a key of row
            if "Info/grdQ/grd_q" in row:
                print(row["Info/grdQ/grd_q"], row["General/timesteps_done"])
                x.append(row["General/timesteps_done"])
                y.append(row["Info/grdQ/grd_q"])

            # try:
            #     print(row["Info/grdQ/grd_q"])
            #     print(row["General/timesteps_done"])
            #     x.append(row["General/timesteps_done"])
            #     y.append(row["Info/grdQ/grd_q"])
            # except:
            #     pass
            # pass
        break

print(len(x), len(y))
