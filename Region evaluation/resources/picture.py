import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']


def plot_leidatu(data, result_dir):
    for village in data['village'].unique().tolist():
        df = data[data['village'] == village]
        df = df.groupby(by=['village', 'aspect'])['Q_TF_IDW'].sum().reset_index()
        name = df['aspect'].tolist()
        angles = np.linspace(0, 2 * np.pi, len(name), endpoint=False)
        value = np.array(df['Q_TF_IDW'].tolist())

        name = np.concatenate([name, [name[0]]])
        angles = np.concatenate((angles, [angles[0]]))
        value = np.concatenate((value, [value[0]]))

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)

        color = ['#FFBE7A', '#FA7F6F', '#2878b5', '#32B897', '#8983BF']
        ax.plot(angles, value, '#999999', linewidth=2, )
        ax.fill(angles, value, facecolor=color[np.argmax(value)], alpha=0.40)

        ax.set_thetagrids(angles * 180 / np.pi, name)

        plt.title(f'{village}',  y=-0.1)

        floor = np.floor(value.min())
        ceil = np.ceil(value.max())
        for i in np.arange(floor, ceil, 0.4):
            ax.plot(angles, [i] * len(name), '--', lw=0.5, color='gray')

        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_yticks([])
        plt.grid(c='gray', linestyle='--', )
        ax.set_theta_zero_location('N')
        plt.savefig(os.path.join(result_dir, f'{village}乡村特色雷达图.png'))
        plt.show()



