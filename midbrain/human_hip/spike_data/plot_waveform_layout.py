
from matplotlib import pyplot as plt


def plot_waveform_layout( neuron_id, sd, axs=None, ylim_margin=0):
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(8,8) )
    #axs.set_title(title, fontsize=12)

    #setting parameters that were previously function variables in Sury's code
    data = sd.neuron_data[0][neuron_id]
    k=neuron_id
    ch=data["channel"]
    position=data["position"]
    temp_chs=list(data["neighbor_channels"])
    temp_pos=data["neighbor_positions"]; templates=data["neighbor_templates"]; filename=f"{id}"; nelec=2
    
    axs.set_title(f"{filename} cluster {k} channel {ch} {position}")
    assert len(temp_chs) == len(temp_pos) == len(templates), "Input length should be the same!"
    nums = len(temp_chs)
    pitch = 17.5
    axs.scatter(position[0], position[1], linewidth=10, alpha=0.2, color='grey')
    axs.text(position[0], position[1], str(position), color="g", fontsize=12)
    # set same scaling to the insets
    index = temp_chs.index(ch)
    ylim_min = min(templates[index])
    ylim_max = max(templates[index])
    # choose channels that are close to the center channel
    for i in range(nums):
        chn_pos = temp_pos[i]
        if position[0] - nelec * pitch <= chn_pos[0] <= position[0] + nelec * pitch \
                and position[1] - nelec * pitch <= chn_pos[1] <= position[1] + nelec * pitch:
            # axs.scatter(chn_pos[0], chn_pos[1], color='w')
            axin = axs.inset_axes([chn_pos[0]-5, chn_pos[1]-5, 15, 20], transform=axs.transData)
            axin.plot(templates[i], color='k', linewidth=2, alpha=0.7)
            axin.set_ylim([ylim_min - ylim_margin, ylim_max + ylim_margin])
            axin.set_axis_off()
    # axs.legend(loc="upper right", fontsize=12)
    # axs.xaxis.set_visible(False)
    # axs.yaxis.set_visible(False)
    axs.set_xlim(position[0]-1.5*nelec*pitch, position[0]+1.5*nelec*pitch)
    axs.set_ylim(position[1]-1.5*nelec*pitch, position[1]+1.5*nelec*pitch)
    axs.invert_yaxis()
    return axs