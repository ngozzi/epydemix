import matplotlib.pyplot as plt

def get_timeseries_data(df_quantiles, compartment, demographic_group, quantile): 
    return df_quantiles.loc[(df_quantiles.compartment == compartment) & \
                            (df_quantiles["quantile"] == quantile) & \
                            (df_quantiles.demographic_group == demographic_group)]

def plot_quantiles(df_quantiles, compartment, demographic_group, ax=None,
                   lower_q=0.05, upper_q=0.95, show_median=True, color="dodgerblue", 
                   ci_alpha=0.3, label="", title="", show_legend=True):
    
    if ax is None:
        fig, ax = plt.subplots(dpi=300, figsize=(10,4))

    if show_median:
        df_med = get_timeseries_data(df_quantiles, compartment, demographic_group, 0.5)
        ax.plot(df_med.date, df_med.value, color=color, label=label)

    df_q1 = get_timeseries_data(df_quantiles, compartment, demographic_group, lower_q)
    df_q2 = get_timeseries_data(df_quantiles, compartment, demographic_group, upper_q)
    ax.fill_between(df_q1.date, df_q1.value, df_q2.value, alpha=ci_alpha, color=color, linewidth=0.)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.3)

    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)