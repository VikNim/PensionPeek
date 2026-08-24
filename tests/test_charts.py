from __future__ import annotations

from pensionpeek.charts import (
    asset_categories_chart,
    balance_sheet_chart,
    net_asset_reconciliation_chart,
)
from pensionpeek.models import FilingMetrics


def test_asset_category_chart_separates_legend_and_hides_tiny_labels() -> None:
    metrics = FilingMetrics(
        asset_categories={
            "Cash": 2.0,
            "Participant loans": 3.0,
            "Common / collective trusts": 700.0,
            "Registered investment companies": 200.0,
            "Other general investments": 95.0,
        }
    )

    figure = asset_categories_chart(metrics)

    assert figure is not None
    assert figure.layout.legend.orientation == "v"
    assert figure.layout.legend.x == 0.62
    assert tuple(figure.data[0].domain.x) == (0, 0.58)
    assert tuple(figure.data[0].text)[:2] == ("", "")


def test_asset_category_chart_groups_long_category_lists() -> None:
    metrics = FilingMetrics(
        asset_categories={f"Category {index}": float(index) for index in range(1, 11)}
    )

    figure = asset_categories_chart(metrics)

    assert figure is not None
    assert len(figure.data[0].labels) == 7
    assert "Other reported categories" in figure.data[0].labels
    assert sum(figure.data[0].values) == sum(metrics.asset_categories.values())


def test_balance_sheet_chart_compares_beginning_and_end_of_year() -> None:
    metrics = FilingMetrics(
        assets_boy=100.0,
        assets_eoy=120.0,
        liabilities_boy=10.0,
        liabilities_eoy=8.0,
        net_assets_boy=90.0,
        net_assets_eoy=112.0,
    )

    figure = balance_sheet_chart(metrics)

    assert figure is not None
    assert [trace.name for trace in figure.data] == ["Beginning of year", "End of year"]
    assert tuple(figure.data[0].y) == ("Assets", "Liabilities", "Net assets")
    assert balance_sheet_chart(FilingMetrics()) is None


def test_net_asset_reconciliation_uses_calculated_residual() -> None:
    metrics = FilingMetrics(
        net_assets_boy=100.0,
        net_assets_eoy=135.0,
        total_income=50.0,
        total_expenses=20.0,
    )

    figure = net_asset_reconciliation_chart(metrics)

    assert figure is not None
    assert tuple(figure.data[0].measure) == ("absolute", "relative", "relative", "total")
    assert tuple(figure.data[0].y) == (100.0, 30.0, 5.0, 135.0)
    assert net_asset_reconciliation_chart(FilingMetrics()) is None
