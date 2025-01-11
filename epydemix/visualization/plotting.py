import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List, Optional, Union, Any


def get_timeseries_data(df_quantiles: pd.DataFrame, 
                        column: str, 
                        quantile: float) -> pd.DataFrame:
    """
    Extracts the time series data for a specific column (compartment or demographic group) and quantile.

    Args:
        df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        column (str): The name of the column to extract data for.
        quantile (float): The quantile to extract data for.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data for the specified column and quantile.
    """
    return df_quantiles.loc[(df_quantiles["quantile"] == quantile)][["date", column]]


def plot_quantiles(results: Any, 
                   columns: Union[List[str], str], 
                   ax: Optional[plt.Axes] = None,
                   lower_q: float = 0.05, 
                   upper_q: float = 0.95, 
                   show_median: bool = True, 
                   ci_alpha: float = 0.3, 
                   title: str = "", 
                   show_legend: bool = True, 
                   palette: str = "Dark2", 
                   colors: Optional[Union[List[str], str]] = None, 
                   labels: Optional[Union[List[str], str]] = None) -> None:
    """
    Plots the quantiles for a specific compartment and demographic group over time.

    Args:
        results (Any): An object containing the simulation results with a `get_df_quantiles` method.
        columns (Union[List[str], str]): The names of the columns to plot data for.
        ax (Optional[plt.Axes], optional): The axes to plot on. If None, a new figure and axes are created (default is None).
        lower_q (float, optional): The lower quantile to plot (default is 0.05).
        upper_q (float, optional): The upper quantile to plot (default is 0.95).
        show_median (bool, optional): Whether to show the median (default is True).
        ci_alpha (float, optional): The alpha value for the confidence interval shading (default is 0.3).
        title (str, optional): The title of the plot (default is an empty string).
        show_legend (bool, optional): Whether to show the legend (default is True).
        palette (str, optional): The color palette for the plot (default is "Set2").
        colors (Optional[Union[List[str], str]], optional): The colors to use for the plot. If None, colors from the palette are used (default is None).
        labels (Optional[Union[List[str], str]], optional): Labels for the lines. If None, column names are used (default is None).
    
    Returns:
        None
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


def plot_selected_quantiles(calibration_results: Any, 
                            ax: Optional[plt.Axes] = None, 
                            show_data: bool = True, 
                            columns: Union[str, List[str]] = "data", 
                            lower_q: float = 0.05, 
                            upper_q: float = 0.95, 
                            show_median: bool = True, 
                            ci_alpha: float = 0.3, 
                            title: str = "", 
                            show_legend: bool = True, 
                            ylabel: str = "", 
                            palette: str = "Dark2") -> None:
    """
    Plots the selected quantiles from the calibration results.

    Args:
        calibration_results (Any): An object containing the calibration results, including selected quantiles and observed data.
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        show_data (bool, optional): Whether to show the observed data points (default is True).
        columns (Union[str, List[str]], optional): The columns to plot from the quantiles data (default is "data").
        lower_q (float, optional): The lower quantile to plot (default is 0.05).
        upper_q (float, optional): The upper quantile to plot (default is 0.95).
        show_median (bool, optional): Whether to show the median line (default is True).
        ci_alpha (float, optional): The alpha value for the confidence interval shading (default is 0.3).
        title (str, optional): The title of the plot (default is an empty string).
        show_legend (bool, optional): Whether to show the legend (default is True).
        ylabel (str, optional): The label for the y-axis (default is an empty string).
        palette (str, optional): The color palette to use for the plot (default is "Set2").

    Returns:
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


def plot_posterior_distribution(calibration_results: Any, 
                                parameter: str, 
                                generation = None,  
                                ax: Optional[plt.Axes] = None, 
                                xlabel: Optional[str] = None, 
                                kind: str = "hist", 
                                color: str = "dodgerblue", 
                                ylabel: str = "", 
                                prior_range: bool = False, 
                                **kwargs) -> None:
    """
    Plots the posterior distribution of a given parameter from the calibration results.

    Args:
        calibration_results (Any): An object containing the calibration results, including the posterior distribution.
        parameter (str): The parameter to plot from the posterior distribution.
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        xlabel (Optional[str], optional): The label for the x-axis (default is None).
        kind (str, optional): The type of plot to generate; options are "hist" for histogram, "kde" for kernel density estimate, and "ecdf" for empirical cumulative distribution function (default is "hist").
        color (str, optional): The color to use for the plot (default is "dodgerblue").
        ylabel (str, optional): The label for the y-axis (default is an empty string).
        prior_range (bool, optional): Whether to set the x-axis limits to the range of the prior distribution (default is False).
        **kwargs: Additional keyword arguments to pass to the seaborn plotting function.

    Returns:
        None: This function does not return any values; it produces a plot.
    """
        
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    df_posterior = calibration_results.get_posterior_distribution(generation=generation)
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


def plot_posterior_distribution_2d(calibration_results: Any, 
                      parameter_x: str, 
                      parameter_y: str, 
                      generation = None,
                      ax: Optional[plt.Axes] = None, 
                      xlabel: Optional[str] = None, 
                      ylabel: Optional[str] = None, 
                      kind: str = "hist", 
                      palette: str = "Blues", 
                      prior_range: bool = False, 
                      title: str = None,  
                      **kwargs) -> None:
    """
    Plots the 2D posterior distribution of two given parameters from the calibration results.

    Args:
        calibration_results (Any): An object containing the calibration results, including the posterior distribution.
        parameter_x (str): The parameter to plot on the x-axis from the posterior distribution.
        parameter_y (str): The parameter to plot on the y-axis from the posterior distribution.
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        xlabel (Optional[str], optional): The label for the x-axis (default is None).
        ylabel (Optional[str], optional): The label for the y-axis (default is None).
        kind (str, optional): The type of plot to generate; options are "hist" for histogram and "kde" for kernel density estimate (default is "hist").
        palette (str, optional): The color palette to use for the plot (default is "Blues").
        prior_range (bool, optional): Whether to set the axis limits to the ranges of the prior distributions (default is False).
        **kwargs: Additional keyword arguments to pass to the seaborn plotting function.

    Returns:
        None: This function does not return any values; it produces a plot.
    """
        
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(6,4))

    df_posterior = calibration_results.get_posterior_distribution(generation=generation)
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
    if title is not None:
        ax.set_title(title)

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
                               palette="Dark2"):
    """
    TODO
    """
    return 0


def plot_contact_matrix(population: Any, 
                        layer_name: str, 
                        ax: Optional[plt.Axes] = None, 
                        title: Optional[str] = None, 
                        show_colorbar: bool = True, 
                        cmap: str = "rocket_r", 
                        vmin: Optional[float] = None, 
                        vmax: Optional[float] = None, 
                        labelsize: int = 10) -> None:
    """
    Plots the contact matrix for a given population layer.

    Args:
        population (Any): An object containing population data, including contact matrices.
        layer_name (str): The name of the contact matrix layer to plot.
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        title (Optional[str], optional): The title of the plot (default is None, which uses the layer name as the title).
        show_colorbar (bool, optional): Whether to display the colorbar (default is True).
        cmap (str, optional): The colormap to use for the plot (default is "rocket_r").
        vmin (Optional[float], optional): The minimum value for the color scale (default is None, which uses the minimum value in the contact matrix).
        vmax (Optional[float], optional): The maximum value for the color scale (default is None, which uses the maximum value in the contact matrix).
        labelsize (int, optional): The font size for axis tick labels (default is 10).

    Returns:
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


def plot_population(population: Any, 
                    ax: Optional[plt.Axes] = None, 
                    title: str = "", 
                    color: str = "dodgerblue", 
                    show_perc: bool = False) -> None:
    """
    Plots the population distribution across demographic groups.

    Args:
        population (Any): An object containing population data, including demographic group names and sizes (`Nk_names` and `Nk`).
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        title (str, optional): The title of the plot (default is an empty string).
        color (str, optional): The color to use for the bars in the plot (default is "dodgerblue").
        show_perc (bool, optional): Whether to show the population as a percentage of the total (default is False).

    Returns:
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


def plot_spectral_radius(epimodel: Any, 
                         ax: Optional[plt.Axes] = None, 
                         title: str = "", 
                         color: str = "dodgerblue", 
                         normalize: bool = False, 
                         show_perc: bool = True, 
                         layer: str = "overall", 
                         show_interventions: bool = True, 
                         interventions_palette: str = "Set2", 
                         interventions_colors: Optional[List[str]] = None) -> None:
    """
    Plots the spectral radius of the contact matrices over time.

    Args:
        epimodel (Any): The EpiModel object containing the contact matrices and interventions.
        ax (Optional[plt.Axes], optional): The matplotlib Axes object to plot on. If not provided, a new figure and axes will be created.
        title (str, optional): The title of the plot (default is an empty string).
        color (str, optional): The color of the plot line (default is "dodgerblue").
        normalize (bool, optional): Whether to normalize the spectral radius by the initial value (default is False).
        show_perc (bool, optional): Whether to show the percentage change in the spectral radius (default is True).
        layer (str, optional): The layer of the contact matrix to plot (default is "overall").
        show_interventions (bool, optional): Whether to show the interventions on the plot (default is True).
        interventions_palette (str, optional): The seaborn color palette to use for the interventions (default is "Set2").
        interventions_colors (Optional[List[str]], optional): A list of colors to use for the interventions. If not provided, the palette will be used (default is None).

    Returns:
        None: This function does not return any values; it produces a plot.
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
  
    
def compute_spectral_radius(m: np.ndarray) -> float:
    """
    Computes the spectral radius of a matrix, which is the largest absolute value of its eigenvalues.

    Args:
        m (np.ndarray): The matrix to compute the spectral radius for.

    Returns:
        float: The spectral radius of the matrix.
    """
    return np.max(np.abs(np.linalg.eigvals(m)))
    

def plot_distance_distribution(calibration_results: Any, 
                               generation = None,
                            ax: Optional[plt.Axes] = None, 
                            xlabel: Optional[str] = None, 
                            kind: str = "hist", 
                            color: str = "dodgerblue", 
                            ylabel: str = "", 
                            **kwargs) -> None:
    """
    Plots the error distribution from the calibration results.

    Args:
        calibration_results (Any): An object containing the calibration results, including the error distribution.
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        xlabel (Optional[str], optional): The label for the x-axis (default is None).
        kind (str, optional): The type of plot to generate; options are "hist" for histogram, "kde" for kernel density estimate, and "ecdf" for empirical cumulative distribution function (default is "hist").
        color (str, optional): The color to use for the plot (default is "dodgerblue").
        ylabel (str, optional): The label for the y-axis (default is an empty string).
        **kwargs: Additional keyword arguments to pass to the seaborn plotting function.

    Returns:
        None: This function does not return any values; it produces a plot.
    """
        
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    errors = calibration_results.get_distances(generation=generation)
    if kind == "hist":
        sns.histplot(data=errors, ax=ax, color=color, **kwargs)
    elif kind == "kde": 
        sns.kdeplot(data=errors, ax=ax, fill=True, color=color, **kwargs)
    elif kind == "ecdf": 
        sns.ecdfplot(data=errors, ax=ax, color=color, **kwargs)
    else: 
        raise ValueError("Unknown kind for plot: %s" % kind)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)


def plot_projections(projections: Any, 
                     calibration_results: Any, 
                     ax: Optional[plt.Axes] = None, 
                     show_data: bool = True, 
                     columns: Union[str, List[str]] = "data", 
                     lower_q: float = 0.05, 
                     upper_q: float = 0.95, 
                     show_median: bool = True, 
                     ci_alpha: float = 0.3, 
                     title: str = "", 
                     show_legend: bool = True, 
                     ylabel: str = "", 
                     palette: str = "Dark2") -> None:
    """
    Plots the projections along with the selected quantiles from the calibration results.

    Args:
        projections (Any): The projection data, typically containing the simulated results with quantiles.
        calibration_results (Any): An object containing the calibration results, including selected quantiles and observed data.
        ax (Optional[plt.Axes], optional): A matplotlib axes object to plot on (default is None, which creates a new figure and axes).
        show_data (bool, optional): Whether to show the observed data points (default is True).
        columns (Union[str, List[str]], optional): The columns to plot from the quantiles data (default is "data").
        lower_q (float, optional): The lower quantile to plot (default is 0.05).
        upper_q (float, optional): The upper quantile to plot (default is 0.95).
        show_median (bool, optional): Whether to show the median line (default is True).
        ci_alpha (float, optional): The alpha value for the confidence interval shading (default is 0.3).
        title (str, optional): The title of the plot (default is an empty string).
        show_legend (bool, optional): Whether to show the legend (default is True).
        ylabel (str, optional): The label for the y-axis (default is an empty string).
        palette (str, optional): The color palette to use for the plot (default is "Set2").

    Returns:
        None: This function does not return any values; it produces a plot.
    """
    
    if not isinstance(columns, list):
        columns = [columns]
    
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    # get data
    calibration_quantiles = calibration_results.get_selected_quantiles()
    data = calibration_results.get_data()

    colors = sns.color_palette(palette, len(columns))
    t = 0

    pleg, handles = [], []
    for column in columns:
        if show_median:
            df_med = get_timeseries_data(projections, column, 0.5)
            p1, = ax.plot(df_med.date, df_med[column].values, color=colors[t])

        df_q1 = get_timeseries_data(projections, column, lower_q)
        df_q2 = get_timeseries_data(projections, column, upper_q)
        p2 = ax.fill_between(df_q1.date, df_q1[column].values, df_q2[column].values, alpha=ci_alpha, color=colors[t], linewidth=0.)
        pleg.append((p1, p2))
        handles.append(f"median ({np.round((1 - lower_q * 2) * 100, 0)}% CI)")
        t += 1

    if show_data: 
        p_actual = ax.scatter(calibration_quantiles.date.unique(), data["data"], s=10, color="k", zorder=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(pleg + [p_actual], handles + ["actual"], loc="upper left", frameon=False)

