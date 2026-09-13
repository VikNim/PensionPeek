from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from planpeek.models import Filing, FilingMetrics

NAVY = "#17324D"
TEAL = "#0D7C7B"
CORAL = "#E36D5D"
GOLD = "#D6A94C"
MINT = "#85B8A5"


def _layout(fig: go.Figure, *, percent: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans Pro, sans-serif", "color": NAVY},
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        height=330,
        hoverlabel={"bgcolor": "white", "font_color": NAVY},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(showgrid=False, linecolor="#D9D4C9")
    fig.update_yaxes(gridcolor="#EAE5DA", tickformat=".0%" if percent else None)
    return fig


def assets_history_chart(history: Sequence[Filing]) -> go.Figure | None:
    points = [(item.plan_year, item.assets_eoy) for item in history]
    points = [(year, value) for year, value in points if year is not None and value is not None]
    if not points:
        return None
    years, assets = zip(*points, strict=True)
    fig = go.Figure(
        go.Scatter(
            x=years,
            y=assets,
            mode="lines+markers",
            line={"color": TEAL, "width": 3},
            marker={"size": 9, "color": TEAL, "line": {"color": "white", "width": 2}},
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
            name="Assets EOY",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(dtick=1)
    return _layout(fig)


def participants_history_chart(
    history: Sequence[Filing],
    selected_year: int | None,
    metrics: FilingMetrics | None,
) -> go.Figure | None:
    years = sorted({item.plan_year for item in history if item.plan_year is not None})
    if not years:
        return None
    boy_by_year = {item.plan_year: item.participants_boy for item in history}
    eoy_by_year: dict[int, int | None] = {year: None for year in years}
    if selected_year is not None and metrics and metrics.participants_eoy is not None:
        eoy_by_year[selected_year] = metrics.participants_eoy
    fig = go.Figure()
    fig.add_bar(
        x=years,
        y=[boy_by_year.get(year) for year in years],
        name="Beginning of year",
        marker_color=TEAL,
        hovertemplate="%{x}: %{y:,.0f}<extra>BOY</extra>",
    )
    if any(value is not None for value in eoy_by_year.values()):
        fig.add_bar(
            x=years,
            y=[eoy_by_year[year] for year in years],
            name="End of year (parsed PDF)",
            marker_color=CORAL,
            hovertemplate="%{x}: %{y:,.0f}<extra>EOY</extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(dtick=1)
    return _layout(fig)


def income_expense_chart(metrics: FilingMetrics) -> go.Figure | None:
    if metrics.total_income is None and metrics.total_expenses is None:
        return None
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    if metrics.total_income is not None:
        labels.append("Income")
        values.append(metrics.total_income)
        colors.append(TEAL)
    if metrics.total_expenses is not None:
        labels.append("Expenses")
        values.append(metrics.total_expenses)
        colors.append(CORAL)
    if metrics.net_income is not None:
        labels.append("Net change")
        values.append(metrics.net_income)
        colors.append(GOLD)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"${value:,.0f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(fig)


def balance_sheet_chart(metrics: FilingMetrics) -> go.Figure | None:
    rows = [
        ("Assets", metrics.assets_boy, metrics.assets_eoy),
        ("Liabilities", metrics.liabilities_boy, metrics.liabilities_eoy),
        ("Net assets", metrics.net_assets_boy, metrics.net_assets_eoy),
    ]
    rows = [row for row in rows if row[1] is not None or row[2] is not None]
    if not rows:
        return None
    labels = [row[0] for row in rows]
    fig = go.Figure()
    if any(row[1] is not None for row in rows):
        fig.add_bar(
            y=labels,
            x=[row[1] for row in rows],
            name="Beginning of year",
            orientation="h",
            marker_color=MINT,
            hovertemplate="%{y}<br>BOY: $%{x:,.0f}<extra></extra>",
        )
    if any(row[2] is not None for row in rows):
        fig.add_bar(
            y=labels,
            x=[row[2] for row in rows],
            name="End of year",
            orientation="h",
            marker_color=TEAL,
            hovertemplate="%{y}<br>EOY: $%{x:,.0f}<extra></extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickprefix="$", tickformat="~s", automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _layout(fig)


def net_asset_reconciliation_chart(metrics: FilingMetrics) -> go.Figure | None:
    if (
        metrics.net_assets_boy is None
        or metrics.net_assets_eoy is None
        or metrics.net_income is None
    ):
        return None
    residual = metrics.net_assets_eoy - metrics.net_assets_boy - metrics.net_income
    fig = go.Figure(
        go.Waterfall(
            x=[
                "Beginning<br>net assets",
                "Net income",
                "Reconciling<br>change",
                "Ending<br>net assets",
            ],
            y=[metrics.net_assets_boy, metrics.net_income, residual, metrics.net_assets_eoy],
            measure=["absolute", "relative", "relative", "total"],
            connector={"line": {"color": "#A8A195", "width": 1}},
            increasing={"marker": {"color": TEAL}},
            decreasing={"marker": {"color": CORAL}},
            totals={"marker": {"color": NAVY}},
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat="~s", automargin=True)
    return _layout(fig)


def asset_categories_chart(metrics: FilingMetrics) -> go.Figure | None:
    values = {name: value for name, value in metrics.asset_categories.items() if value > 0}
    if not values:
        return None
    if len(values) > 7:
        ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
        values = dict(ranked[:6])
        values["Other reported categories"] = sum(value for _name, value in ranked[6:])
    total = sum(values.values())
    labels = list(values)
    percentages = [value / total for value in values.values()]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=list(values.values()),
            hole=0.62,
            sort=False,
            domain={"x": [0, 0.58], "y": [0.04, 0.96]},
            marker={"colors": [TEAL, CORAL, GOLD, MINT, NAVY, "#A48CB4", "#7895B2"]},
            text=[f"{percentage:.1%}" if percentage >= 0.03 else "" for percentage in percentages],
            textinfo="text",
            textposition="inside",
            insidetextorientation="horizontal",
            hovertemplate="%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        )
    )
    fig = _layout(fig)
    fig.update_layout(
        height=max(360, 32 * len(labels) + 160),
        margin={"l": 5, "r": 5, "t": 10, "b": 10},
        legend={
            "orientation": "v",
            "x": 0.62,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "font": {"size": 11},
        },
        uniformtext={"minsize": 10, "mode": "hide"},
        annotations=[
            {
                "text": "Plan-level<br>categories",
                "x": 0.29,
                "y": 0.5,
                "font": {"size": 14, "color": NAVY},
                "showarrow": False,
            }
        ],
    )
    return fig
