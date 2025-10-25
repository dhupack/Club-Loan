
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import os

def save_and_display(fig, filename="figure.png", dpi=150):
    """Save a matplotlib figure and display it (works in notebooks)."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    try:
        from IPython.display import Image, display
        display(Image(filename))
    except Exception:
        print(f"Saved to {filename}")

def _gradient_colors(start=(0.2,0.4,0.8), end=(1,1,1), n=100):
    start = np.array(start)
    end = np.array(end)
    return [tuple(start + (end-start) * (i/(n-1))) for i in range(n)]

def plot_gradient_bar(values, labels=None, title=None, figsize=(8,4)):
    """Simple vertical gradient bars using matplotlib."""
    if labels is None:
        labels = [str(i) for i in range(len(values))]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(values))
    maxh = max(values) if len(values) else 1
    for xi, val in zip(x, values):
        grad = np.vstack([np.linspace(0.2,1,256),]*3).T  # greyscale gradient
        ax.imshow(grad, extent=(xi-0.4, xi+0.4, 0, val), aspect='auto', zorder=1)
        ax.bar(xi, val, width=0.8, alpha=0.0, zorder=2)  # invisible bar to keep axes scaled
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, maxh*1.05)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_gradient_bar_grouped(df, category_col, value_col, group_col=None, title=None, figsize=(8,4)):
    """
    df: pandas DataFrame
    category_col: x-axis categories
    value_col: numeric value
    group_col: optional grouping column (stacked side-by-side groups)
    """
    import pandas as pd
    grouped = df.copy()
    if group_col is None:
        agg = grouped.groupby(category_col)[value_col].mean().reset_index()
        return plot_gradient_bar(agg[value_col].values, labels=agg[category_col].astype(str).tolist(), title=title, figsize=figsize)
    else:
        # simple grouped bar: pivot
        pivot = grouped.pivot_table(index=category_col, columns=group_col, values=value_col, aggfunc='mean').fillna(0)
        labels = pivot.index.astype(str).tolist()
        x = np.arange(len(labels))
        width = 0.8 / max(1, pivot.shape[1])
        fig, ax = plt.subplots(figsize=figsize)
        for i, col in enumerate(pivot.columns):
            vals = pivot[col].values
            ax.bar(x + (i - (pivot.shape[1]-1)/2) * width, vals, width=width, label=str(col))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        if title: ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig

def plot_woe_distribution_plotly(df, feature, target, bins=10, title=None):
    """
    Simple WOE-like plot: computes binned event rate and plots log-odds.
    df: DataFrame containing 'feature' and binary 'target' (0/1).
    """
    import pandas as pd
    ser = df[feature]
    try:
        df2 = df[[feature, target]].copy()
        df2['bin'] = pd.qcut(df2[feature].rank(method='first'), q=bins, duplicates='drop')
    except Exception:
        # fallback: equal-width bins
        df2 = df[[feature, target]].copy()
        df2['bin'] = pd.cut(df2[feature], bins=bins)
    agg = df2.groupby('bin')[target].agg(['sum','count']).reset_index()
    agg['non_event'] = agg['count'] - agg['sum']
    # avoid division by zero
    agg['event_rate'] = (agg['sum'] + 0.5) / (agg['count'] + 1)
    agg['woe'] = np.log(agg['event_rate'] / (1 - agg['event_rate']))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg['bin'].astype(str), y=agg['count'], name='count', yaxis='y1', opacity=0.6))
    fig.add_trace(go.Scatter(x=agg['bin'].astype(str), y=agg['woe'], name='WOE', yaxis='y2', mode='lines+markers'))
    fig.update_layout(
        title=title or f"WOE-like plot for {feature}",
        xaxis_tickangle=-45,
        yaxis=dict(title='count'),
        yaxis2=dict(title='WOE (log-odds)', overlaying='y', side='right')
    )
    return fig

def plot_stacked_categorical_gradient(df, category_col, stacked_cols, title=None, figsize=(8,4)):
    """
    df: DataFrame
    category_col: x-axis categories
    stacked_cols: list of columns to stack (proportions)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    pivot = df.groupby(category_col)[stacked_cols].sum()
    proportions = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    x = np.arange(len(proportions))
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(proportions))
    for col in proportions.columns:
        vals = proportions[col].values
        ax.bar(x, vals, bottom=bottom, label=str(col))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(proportions.index.astype(str), rotation=45, ha='right')
    if title: ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    return fig

