import matplotlib.pyplot as plt
import numpy as np

def get_colors_from_palette(palette_name, n_colors):
    # Retrieve the colormap
    cmap = plt.get_cmap(palette_name)
    # Generate N evenly spaced values between 0 and 1
    color_indices = np.linspace(0, 1, n_colors)
    # Generate the colors
    colors = [cmap(index) for index in color_indices]
    np.random.shuffle(colors)
    return colors


def get_timeseries_data(df_quantiles, compartment, demographic_group, quantile): 
    """
    Extracts the time series data for a specific compartment, demographic group, and quantile.

    Parameters:
    -----------
        - df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        - compartment (str): The name of the compartment to extract data for.
        - demographic_group (str or int): The demographic group to extract data for.
        - quantile (float): The quantile to extract data for.

    Returns:
    --------
        - pd.DataFrame: A DataFrame containing the time series data for the specified compartment, demographic group, and quantile.
    """
    return df_quantiles.loc[(df_quantiles.compartment == compartment) & \
                            (df_quantiles["quantile"] == quantile) & \
                            (df_quantiles.demographic_group == demographic_group)]

def plot_quantiles(df_quantiles, compartments, demographic_groups, ax=None,
                   lower_q=0.05, upper_q=0.95, show_median=True, 
                   ci_alpha=0.3, title="", show_legend=True, 
                   palette="Set2"):
    """
    Plots the quantiles for a specific compartment and demographic group over time.

    Parameters:
    -----------
        - df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        - compartment (list or str): The names of the compartment to plot data for.
        - demographic_group (list or str or int): The demographic groups to plot data for.
        - ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
        - lower_q (float, optional): The lower quantile to plot (default is 0.05).
        - upper_q (float, optional): The upper quantile to plot (default is 0.95).
        - show_median (bool, optional): Whether to show the median (default is True).
        - ci_alpha (float, optional): The alpha value for the confidence interval shading (default is 0.3).
        - label (str, optional): The label for the median line (default is "").
        - title (str, optional): The title of the plot (default is "").
        - show_legend (bool, optional): Whether to show legend (default is True).
        - palette (str, optional): The color palette for the plot (default is "Set2")
    """
    
    if not isinstance(compartments, list):
        compartments = [compartments]

    if not isinstance(demographic_groups, list):
        demographic_groups = [demographic_groups]

    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    colors = get_colors_from_palette(palette, len(compartments) * len(demographic_groups))
    t = 0
    for compartment in compartments:
        for group in demographic_groups:
            if show_median:
                df_med = get_timeseries_data(df_quantiles, compartment, group, 0.5)
                ax.plot(df_med.date, df_med.value, color=colors[t], label=f"{compartment}_{group}")

            df_q1 = get_timeseries_data(df_quantiles, compartment, group, lower_q)
            df_q2 = get_timeseries_data(df_quantiles, compartment, group, upper_q)
            ax.fill_between(df_q1.date, df_q1.value, df_q2.value, alpha=ci_alpha, color=colors[t], linewidth=0.)
            t += 1

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)

    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)