import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.stats import zscore

# calculates weighted volume from volume and density
def weighted_volume(data):
    '''
    calculates weighted volume from volume and density
    requires at least 4 points for ConvexHull to compute volume
    avoids division by zero by adding a small constant to volume
    returns weighted volume and density
    '''
    if len(data) >= 4:  # needs at least 4 points for ConvexHull
        hull = ConvexHull(data)
        volume = hull.volume
    else:
        volume = 0
    density = len(data) / (volume + 1e-10)  # avoid division by zero
    weighted_vol = volume * density  # weight volume by density
    return weighted_vol, density

# removes outliers using z-score method
def outliers_z_method(data, threshold=3):
    '''
    removes outliers using z-score method
    applies threshold to filter outliers based on z-score
    returns data with outliers removed
    '''
    z_scores = np.abs(zscore(data, axis=0))
    filtered_indices = (z_scores < threshold).all(axis=1)
    return data[filtered_indices]

# visualizes DBSCAN clustering with weighted volume calculation
def dbscan_advanced_visualize_weighted(time, data_list, eps=3, min_samples=5, show_removed_points=False, title="":
    '''
    visualizes DBSCAN clustering with weighted volume calculation
    allows for dynamic eps and min_samples for DBSCAN
    optionally shows removed points
    supports 3D visualization with matplotlib
    returns cluster volumes, densities, and position-velocity labels
    '''
    x, y, z, vx, vy, vz = [], [], [], [], [], []
    for data in data_list:
        time_data = data[data[:, 0] == time]
        if time_data.size != 0:
            x.append(time_data[0, 1])
            y.append(time_data[0, 2])
            z.append(time_data[0, 3])
            vx.append(time_data[0, 4])
            vy.append(time_data[0, 5])
            vz.append(time_data[0, 6])
    pos_vel = np.column_stack((x, y, z, vx, vy, vz))

    results = {"clusters": {}, "stars": None}  # store results

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pos_vel)
    labels = db.labels_
    pos_vel_labels = np.column_stack((pos_vel, labels))

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    core_samples = pos_vel[core_samples_mask]
    noise_samples = pos_vel[~core_samples_mask]

    if noise_samples.size == 0:
        noise_labels = np.array([])
    else:
        db_noise = DBSCAN(eps=2, min_samples=5).fit(noise_samples[:, :3])
        noise_labels = db_noise.labels_

    # figure and axes setup
    plt.rcParams['figure.dpi'] = 600
    bg_color = "#01072C"
    text_color = "white"
    grid_color = "#2E2D3D"
    CB_color_cycle = ['#6D0BFF', '#00D0FF', '#0069FF', '#4DA3FF', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    color_palette = sns.color_palette(CB_color_cycle, len(set(labels)))

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.grid(True)

    # axes and tick label styling
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_color(text_color)

    # plot core samples and noise clusters
    ax.scatter(core_samples[:, 0], core_samples[:, 1], core_samples[:, 2], c='white', s=2, label='Core Samples', alpha=0.2)
    legend_labels = ['Core Samples']
    cluster_info_texts = []
    unique_noise_labels = set(noise_labels)
    if -1 in unique_noise_labels:
        unique_noise_labels.remove(-1)

    cluster_volumes = []
    cluster_densities = []
    for i, label in enumerate(unique_noise_labels):
        subset = noise_samples[noise_labels == label]
        subset = outliers_z_method(subset)
        if subset.ndim == 2 and subset.size != 0:
            color = color_palette[i] if i < len(color_palette) else 'gray'
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=[color], s=10, marker='o', alpha=1)
            volume, density = weighted_volume(subset[:, :3])
            cluster_volumes.append(volume)
            cluster_densities.append(density)
            cluster_info_text = f"Cluster {label + 1}:\nPoints: {len(subset)}\nVolume: {volume:.2f}\nDensity: {density:.2f}"
            cluster_info_texts.append((cluster_info_text, color))
            legend_labels.append(f'Noise Cluster {label + 1}')

    # axis labels and legend
    ax.set_xlabel('X Position', color=text_color)
    ax.set_ylabel('Y Position', color=text_color)
    ax.set_zlabel('Z Position', color=text_color)
    lgd = ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.35, 0.41), framealpha=0.0)
    for text in lgd.get_texts():
        text.set_color(text_color)

    # display DBSCAN parameters and cluster info
    text_offset = 0.8
    db_parameters = f"$\epsilon = $ {db.eps}\n$minPts = ${db.min_samples}"
    ax.text2D(1.15, text_offset, db_parameters, transform=ax.transAxes, va='top', ha='left', color=text_color, fontsize=10)
    for cluster_info_text, color in cluster_info_texts:
        text_offset -= 0.04 * (cluster_info_text.count('\n') + 1)
        ax.text2D(1.15, text_offset, cluster_info_text, transform=ax.transAxes, va='top', ha='left', color=color, fontsize=10)

    plt.show()

    return cluster_volumes, cluster_densities, pos_vel_labels