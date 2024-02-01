import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib as mpl


def calculate_gap_significance_score(gap_width, gap_depth):
    '''calculates gap significance score'''
    return gap_width * (1-gap_depth)


def analyze_stream_gaps(stream_data, unperturbed_data, bin_width=5, bin_start=-130, bin_end=50, drop_threshold=0.7, search_area=None, name=None, legendloc=0.02):
    '''analyzes stream gaps, comparing stream data to unperturbed baseline'''

    # color settings
    BAR_COLOR = '#8D98F0'
    GAP_COLOR = '#53F1FA'
    NON_GAP_COLOR = BAR_COLOR
    NON_SIGNIFICANT_GAP_COLOR = '#6D49E1'
    BRIDGE_COLOR = '#3997EA'
    SEARCH_AREA_COLOR = 'white'
    TEXT_COLOR = 'white'
    BACKGROUND_COLOR = '#01072A'
    GRID_COLOR = "#2E2D3D"

    # define dynamic bins based on new parameters
    bin_range = np.arange(bin_start, bin_end + bin_width, bin_width)  # ensure bin_end is inclusive

    # binning data and calculating densities
    density_perturbed, _ = np.histogram(stream_data['phi1'], bins=bin_range)
    density_unperturbed, _ = np.histogram(unperturbed_data['phi1'], bins=bin_range)

    # calculating the ratio of densities
    with np.errstate(divide='ignore', invalid='ignore'):
        density_ratio = np.where(density_unperturbed > 0, density_perturbed / density_unperturbed, np.nan)

    # plotting
    plt.figure(figsize=(14, 7), dpi = 600)
    bars = plt.bar(bin_range[:-1], density_ratio, width=bin_width * 0.9, color=BAR_COLOR, align='edge', edgecolor=BAR_COLOR)
    plt.xlabel('$\phi_1(deg)$', color='black', fontsize=18)
    plt.ylabel('Relative Density Change', color='black', fontsize=18)
    plt.grid(True, color=GRID_COLOR)

    # mark the search area boundaries if provided
    if search_area:
        line_start = plt.axvline(x=search_area[0], color=SEARCH_AREA_COLOR, linestyle='--', linewidth=2)
        line_end = plt.axvline(x=search_area[1], color=SEARCH_AREA_COLOR, linestyle='--', linewidth=2)

        # create custom legend entries
        custom_lines = [Line2D([0], [0], color=SEARCH_AREA_COLOR, linestyle='--', lw=2),
                        Line2D([0], [0], color=GAP_COLOR, lw=4),
                        Line2D([0], [0], color=NON_GAP_COLOR, lw=4),
                        Line2D([0], [0], color=NON_SIGNIFICANT_GAP_COLOR, lw=4),
                        Line2D([0], [0], color=BRIDGE_COLOR, lw=4)]

        # add legend to the plot
        legend = plt.legend(custom_lines, ['Search Area Border', 'Detected Gap', 'Non-Gap', 'Isolation Bar', 'Bridge Bar'], loc='lower left', bbox_to_anchor=(0.1, legendloc))

        # set the background color, outline color, and font color
        legend.get_frame().set_facecolor(BACKGROUND_COLOR)  # dark background
        legend.get_frame().set_edgecolor(TEXT_COLOR)    # white outline
        plt.setp(legend.get_texts(), color=TEXT_COLOR, size = 14)  # white font color

    # identifying gaps within the search area
    gap_info_text = ""
    gap_positions = []
    gap_results = []
    gap_indices = (density_ratio < drop_threshold) & (bin_range[:-1] >= search_area[0]) & (bin_range[:-1] <= search_area[1])
    is_in_gap = False
    start_idx = None
    bridge_idx = None

    for idx, bar in enumerate(bars):
        # apply search area constraints
        if search_area and (bin_range[idx] < search_area[0] or bin_range[idx] > search_area[1]):
            continue  # skip processing if outside the search area

        if np.isnan(density_ratio[idx]) or density_ratio[idx] == 0:  # check for NaN or zero values
            # determine the color of the line based on the gap context
            line_color = GAP_COLOR if gap_indices[idx] else BAR_COLOR
            if start_idx is not None and idx <= end_idx:  # check if within a previously identified non-significant region
                line_color = NON_SIGNIFICANT_GAP_COLOR
            if bridge_idx is not None and idx == bridge_idx:  # check for a bridge
                line_color = BRIDGE_COLOR
            plt.hlines(y=0, xmin=bin_range[idx], xmax=bin_range[idx]+bin_width, colors=line_color, linewidth=5)  # draw a thicker horizontal line

        if gap_indices[idx]:
            bar.set_color(GAP_COLOR)  # color gap bars in pink
            is_in_gap = True
            if start_idx is None:
                start_idx = idx
            if bridge_idx is not None:
                bars[bridge_idx].set_color(BRIDGE_COLOR)  # purple for bridge bar
                bridge_idx = None
        elif is_in_gap:
            # Check for a bridge bar scenario
            if idx + 1 < len(gap_indices) and gap_indices[idx + 1]:
                bridge_idx = idx
            else:
                end_idx = idx if bridge_idx is None else bridge_idx + 1
                gap_width = bin_range[end_idx] - bin_range[start_idx]
                if gap_width >= max(3, 2 * bin_width):
                    gap_density_values = density_ratio[start_idx:end_idx]
                    gap_depth = np.nanmean(gap_density_values)
                    num_stars = np.sum(density_perturbed[start_idx:end_idx])  # number of stars in the gap

                    gap_significance_score = calculate_gap_significance_score(gap_width, gap_depth)
                    position_range = f"({bin_range[start_idx]:.2f}, {bin_range[end_idx]:.2f})"
                    gap_results.append({
                        'width': gap_width,
                        'depth': gap_depth,
                        'GSS': gap_significance_score,
                        'position': position_range
                    })
                    gap_info_text += (f'Gap Score: {gap_significance_score:.2f},  '
                                      f'Gap Width: {gap_width:.2f},  '
                                      f'Gap Depth: {gap_depth:.2f},  '
                                      f'Position: {position_range}\n\n')
                    gap_positions.append(position_range)
                else:
                    for bar_idx in range(start_idx, end_idx):
                        bars[bar_idx].set_color(NON_SIGNIFICANT_GAP_COLOR)  # color bars that don't meet criteria in light purple
                is_in_gap = False
                start_idx = None
                bridge_idx = None

    plt.title(gap_info_text.strip(), fontsize=18 )
    plt.tight_layout(pad=2) 

    plt.gca().set_facecolor(BACKGROUND_COLOR)
    plt.gca().tick_params(axis='both', which='major', labelsize=16, length=6, color='black')

    sns.set_style('darkgrid')
    plt.show()

    return gap_results