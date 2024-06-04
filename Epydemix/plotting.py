import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_timeseries_data(df_quantiles, column, quantile): 
    """
    Extracts the time series data for a specific compartment, demographic group, and quantile.

    Parameters:
    -----------
        - df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        - column (str): The name of the column to extract data for.
        - quantile (float): The quantile to extract data for.

    Returns:
    --------
        - pd.DataFrame: A DataFrame containing the time series data for the specified compartment, demographic group, and quantile.
    """
    return df_quantiles.loc[(df_quantiles["quantile"] == quantile)][["date", column]]


def plot_quantiles(results, columns, ax=None,
                   lower_q=0.05, upper_q=0.95, show_median=True, 
                   ci_alpha=0.3, title="", show_legend=True, 
                   palette="Set2", colors=None, labels=None):
    """
    Plots the quantiles for a specific compartment and demographic group over time.

    Parameters:
    -----------
        - results (SimulationResults): An object containing the simulation results
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
        - colors (list or str, optional): The colors to use for the plot (default is None).
    """
    
    df_quantiles = results.get_df_quantiles()
    
    if not isinstance(columns, list):
        columns = [columns]

    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    if colors is None:
        colors = sns.color_palette(palette, len(columns))
    else: 
        if not isinstance(colors, list):
            colors = [colors]

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels]
    else: 
        labels = columns

    t = 0
    for column in columns:
        if show_median:
            df_med = get_timeseries_data(df_quantiles, column, 0.5)
            ax.plot(df_med.date, df_med[column].values, color=colors[t], label=labels[t])

        df_q1 = get_timeseries_data(df_quantiles, column, lower_q)
        df_q2 = get_timeseries_data(df_quantiles, column, upper_q)
        ax.fill_between(df_q1.date, df_q1[column].values, df_q2[column].values, alpha=ci_alpha, color=colors[t], linewidth=0.)
        t += 1

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)

    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)


def plot_selected_quantiles(calibration_results, ax=None, show_data=True, columns="data", 
                               lower_q=0.05, upper_q=0.95, show_median=True, 
                               ci_alpha=0.3, title="", show_legend=True, ylabel="", 
                               palette="Set2"):
    
    """
    Plots the selected quantiles from the calibration results.

    Parameters:
    -----------
        calibration_results (CalibrationResults): An object containing the calibration results, including selected quantiles and observed data.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        show_data (bool, optional): Whether to show the observed data points (default is True).
        columns (str or list of str, optional): The columns to plot from the quantiles data (default is "data").
        lower_q (float, optional): The lower quantile to plot (default is 0.05).
        upper_q (float, optional): The upper quantile to plot (default is 0.95).
        show_median (bool, optional): Whether to show the median line (default is True).
        ci_alpha (float, optional): The alpha value for the confidence interval shading (default is 0.3).
        title (str, optional): The title of the plot (default is "").
        show_legend (bool, optional): Whether to show the legend (default is True).
        ylabel (str, optional): The label for the y-axis (default is "").
        palette (str, optional): The color palette to use for the plot (default is "Set2").

    Returns:
    --------
        None: This function does not return any values; it produces a plot.
    """
    
    if not isinstance(columns, list):
        columns = [columns]
    
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    # get selected quantiles and data
    df_quantiles = calibration_results.get_selected_quantiles()
    data = calibration_results.get_data()

    colors = sns.color_palette(palette, len(columns))
    t = 0

    pleg, handles = [], []
    for column in columns:
        if show_median:
            df_med = get_timeseries_data(df_quantiles, column, 0.5)
            p1, = ax.plot(df_med.date, df_med[column].values, color=colors[t])

        df_q1 = get_timeseries_data(df_quantiles, column, lower_q)
        df_q2 = get_timeseries_data(df_quantiles, column, upper_q)
        p2 = ax.fill_between(df_q1.date, df_q1[column].values, df_q2[column].values, alpha=ci_alpha, color=colors[t], linewidth=0.)
        pleg.append((p1, p2))
        handles.append(f"median ({np.round((1 - lower_q * 2) * 100, 0)}% CI)")
        t += 1

    if show_data: 
        p_actual = ax.scatter(df_q1.date, data["data"], s=10, color="k", zorder=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(pleg + [p_actual], handles + ["actual"], loc="upper left", frameon=False)


def plot_posterior(calibration_results, parameter, ax=None, xlabel=None, kind="hist", color="dodgerblue", ylabel="", prior_range=False, **kwargs): 
    """
    Plots the posterior distribution of a given parameter from the calibration results.

    Parameters:
    -----------
        calibration_results (CalibrationResults): An object containing the calibration results, including the posterior distribution.
        parameter (str): The parameter to plot from the posterior distribution.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        xlabel (str, optional): The label for the x-axis (default is None).
        kind (str, optional): The type of plot to generate; options are "hist" for histogram, "kde" for kernel density estimate, and "ecdf" for empirical cumulative distribution function (default is "hist").
        color (str, optional): The color to use for the plot (default is "dodgerblue").
        ylabel (str, optional): The label for the y-axis (default is "").
        prior_range (bool, optional): Whether to set the x-axis limits to the range of the prior distribution (default is False).
        **kwargs: Additional keyword arguments to pass to the seaborn plotting function.

    Returns:
    --------
        None: This function does not return any values; it produces a plot.
    """
        
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    df_posterior = calibration_results.get_posterior_distribution()
    if kind == "hist":
        sns.histplot(data=df_posterior, x=parameter, ax=ax, color=color, **kwargs)
    elif kind == "kde": 
        sns.kdeplot(data=df_posterior, x=parameter, ax=ax, fill=True, color=color, **kwargs)
    elif kind == "ecdf": 
        sns.ecdfplot(data=df_posterior, x=parameter, ax=ax, color=color, **kwargs)
    else: 
        raise ValueError("Unknown kind for plot: %s" % kind)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)

    if prior_range: 
        xmin, xmax = calibration_results.get_priors()[parameter].ppf(0), calibration_results.get_priors()[parameter].ppf(1)
        ax.set_xlim(xmin, xmax)


def plot_posterior_2d(calibration_results, parameter_x, parameter_y, ax=None, xlabel=None, ylabel=None, kind="hist", palette="Blues", prior_range=False, **kwargs): 
    """
    Plots the 2D posterior distribution of two given parameters from the calibration results.

    Parameters:
    -----------
        calibration_results (CalibrationResults): An object containing the calibration results, including the posterior distribution.
        parameter_x (str): The parameter to plot on the x-axis from the posterior distribution.
        parameter_y (str): The parameter to plot on the y-axis from the posterior distribution.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        xlabel (str, optional): The label for the x-axis (default is None).
        ylabel (str, optional): The label for the y-axis (default is None).
        kind (str, optional): The type of plot to generate; options are "hist" for histogram and "kde" for kernel density estimate (default is "hist").
        palette (str, optional): The color palette to use for the plot (default is "Blues").
        prior_range (bool, optional): Whether to set the axis limits to the ranges of the prior distributions (default is False).
        **kwargs: Additional keyword arguments to pass to the seaborn plotting function.

    Returns:
    --------
        None: This function does not return any values; it produces a plot.
    """
        
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(6,4))

    df_posterior = calibration_results.get_posterior_distribution()
    if kind == "hist":
        sns.histplot(data=df_posterior, x=parameter_x, y=parameter_y, ax=ax, palette=palette, **kwargs)
    elif kind == "kde": 
        sns.kdeplot(data=df_posterior, x=parameter_x, y=parameter_y, ax=ax, fill=True, palette=palette, **kwargs)
    else: 
        raise ValueError("Unknown kind for plot: %s" % kind)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if prior_range: 
        xmin, xmax = calibration_results.get_priors()[parameter_x].ppf(0), calibration_results.get_priors()[parameter_x].ppf(1)
        ymin, ymax = calibration_results.get_priors()[parameter_y].ppf(0), calibration_results.get_priors()[parameter_y].ppf(1)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


def plot_selected_trajectories(calibration_results, ax=None, show_data=True, columns="data", 
                               lower_q=0.05, upper_q=0.95, show_median=True, 
                               ci_alpha=0.3, title="", show_legend=True, ylabel="", 
                               palette="Set2"):
    """
    TODO
    """
    return 0


def plot_contact_matrix(population, layer_name, ax=None, title=None, show_colorbar=True, cmap="rocket_r", vmin=None, vmax=None, labelsize=10): 
    """
    Plots the contact matrix for a given population layer.

    Parameters:
    -----------
        population (Population): An object containing population data, including contact matrices.
        layer_name (str): The name of the contact matrix layer to plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        title (str, optional): The title of the plot (default is None, which uses the layer name as the title).
        show_colorbar (bool, optional): Whether to display the colorbar (default is True).
        cmap (str, optional): The colormap to use for the plot (default is "rocket_r").
        vmin (float, optional): The minimum value for the color scale (default is None, which uses the minimum value in the contact matrix).
        vmax (float, optional): The maximum value for the color scale (default is None, which uses the maximum value in the contact matrix).

    Returns:
    --------
        None: This function does not return any values; it produces a plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6), dpi=300)

    if vmin is None: 
        vmin = np.min(population.contact_matrices[layer_name])
    if vmax is None: 
        vmax = np.max(population.contact_matrices[layer_name])

    im = ax.imshow(population.contact_matrices[layer_name], origin="lower", cmap=sns.color_palette(cmap, as_cmap=True), vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(population.Nk_names.shape[0]))
    ax.set_yticks(range(population.Nk_names.shape[0]))
    ax.set_xticklabels(population.Nk_names, rotation=90, fontsize=labelsize)
    ax.set_yticklabels(population.Nk_names, fontsize=labelsize)

    if show_colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8)

    if title is None: 
        ax.set_title(layer_name)
    else:
        ax.set_title(title)


def plot_population(population, ax=None, title="", color="dodgerblue", show_perc=False):
    """
    Plots the population distribution across demographic groups.

    Parameters:
    -----------
        population (Population): An object containing population data, including demographic group names and sizes.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        title (str, optional): The title of the plot (default is "").
        color (str, optional): The color to use for the bars in the plot (default is "dodgerblue").
        show_perc (bool, optional): Whether to show the population as a percentage of the total (default is False).

    Returns:
    --------
        None: This function does not return any values; it produces a plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    if show_perc: 
        ax.bar(population.Nk_names, 100 * population.Nk / np.sum(population.Nk), color=color)
        ax.set_ylabel("% of individuals")
    else: 
        ax.bar(population.Nk_names, population.Nk, color=color)
        ax.set_ylabel("Number of individuals")

    ax.tick_params(axis="x", rotation=90)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)
    ax.set_title(title)


def plot_spectral_radius(epimodel, ax=None, title="", color="dodgerblue", normalize=False, show_perc=True, layer="overall", 
                         show_interventions=True, interventions_palette="Set2", interventions_colors=None): 
    """
    Plots the spectral radius of the contact matrices over time.

    Parameters:
    - epimodel: The EpiModel object containing the contact matrices.
    - ax: Optional. The matplotlib Axes object to plot on. If not provided, a new figure and axes will be created.
    - title: Optional. The title of the plot.
    - color: Optional. The color of the plot line.
    - normalize: Optional. Whether to normalize the spectral radius by the initial value.
    - show_perc: Optional. Whether to show the percentage change in the spectral radius.
    - layer: Optional. The layer of the contact matrix to plot.
    - show_interventions: Optional. Whether to show the interventions on the plot.
    - interventions_palette: Optional. The seaborn color palette to use for the interventions.
    - interventions_colors: Optional. The color of the interventions. If not provided, the palette will be used.

    Returns:
    - None

    Raises:
    - None

    Notes:
    - This function requires the EpiModel object to have the contact matrices defined.

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6), dpi=300)

    # compute spectral radius
    if len(epimodel.Cs) == 0:
        print("Contacts over time have not been defined")
        return None

    dates = list(epimodel.Cs.keys())
    rho = [compute_spectral_radius(epimodel.Cs[date][layer]) for date in dates]
    if normalize:
        rho = np.array(rho) / rho[0]
        ylabel = f"$\\rho(C(t)) / \\rho(C(t_0))$ ({layer})"

    elif show_perc:
        rho = (np.array(rho) / rho[0] - 1) * 100
        ylabel = f"$\\rho(C(t))$ % change ({layer})"
        
    else: 
       ylabel = "$\\rho(C(t))$ ({layer})"

    ax.plot(dates, rho, color=color)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if show_interventions:
        if interventions_colors is None:
            interventions_colors = sns.color_palette(interventions_palette, len(epimodel.interventions))
        else: 
            if not isinstance(interventions_colors, list):
                interventions_colors = [interventions_colors]

        y1, y2 = ax.get_ylim()
        x1, x2 = ax.get_xlim()
        for color, interventions in zip(interventions_colors, epimodel.interventions): 
            ax.fill_betweenx([y1, y2], interventions["start_date"], interventions["end_date"], color=color, alpha=0.3, linewidth=0, label=interventions["name"])

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)

        ax.legend(loc="upper right")
  
    
def compute_spectral_radius(m): 
    """
    Computes the spectral radius of a matrix.

    Parameters:
    -----------
        m (np.array): The matrix to compute the spectral radius for.

    Returns:
    --------
        float: The spectral radius of the matrix.
    """
    return np.max(np.abs(np.linalg.eigvals(m)))
    