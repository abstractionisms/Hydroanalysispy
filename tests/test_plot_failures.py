import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrology.visualization.plots import plot_threshold_exceedance


def test_threshold_exceedance_uses_data_driven_thresholds_without_hardcoded_flows():
    dates = pd.date_range("2020-06-01", "2022-09-30", freq="D", tz="UTC")
    summer = dates.month.isin([6, 7, 8, 9])
    values = np.where(summer, np.linspace(20, 120, len(dates)), 500)
    df = pd.DataFrame({"Discharge_cfs": values}, index=dates)

    fig, ax = plt.subplots()
    plot_threshold_exceedance(ax, df_q=df)

    assert "summer flow quantiles" in ax.get_title()
    assert "1000" not in [text.get_text() for text in ax.get_legend().get_texts()]
    plt.close(fig)
