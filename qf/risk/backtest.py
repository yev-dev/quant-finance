import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from qf.risk import DATA_DIR

plt.ioff()


class Backtest:
    def __init__(self, actual, forecast, alpha):
        self.index = actual.index
        self.actual = actual.values
        self.forecast = forecast.values
        self.alpha = alpha

    def hit_series(self):
        return (self.actual < self.forecast) * 1

    def number_of_hits(self):
        return self.hit_series().sum()

    def hit_rate(self):
        return self.hit_series().mean()

    def expected_hits(self):
        return self.actual.size * self.alpha

    def duration_series(self):
        hit_series = self.hit_series()
        hit_series[0] = 1
        hit_series[-1] = 1
        return np.diff(np.where(hit_series == 1))[0]

    def plot(self, file_name=None):

        # Re-add the time series index
        actuals = pd.Series(self.actual, index=self.index)
        forecasted = pd.Series(self.forecast, index=self.index)

        sns.set_context("paper")
        sns.set_style("whitegrid", {"font.family": "serif", "font.serif": "Computer Modern Roman", "text.usetex": True})

        # Hits
        ax = actuals[actuals <= forecasted].plot(color="red", marker="o", ls="None", figsize=(6, 3.5))
        for h in actuals[actuals <= forecasted].index:
            plt.axvline(h, color="black", alpha=0.4, linewidth=1, zorder=0)

        # Positive returns
        actuals[forecasted < actuals].plot(ax=ax, color="green", marker="o", ls="None")

        # Negative returns but no hit
        actuals[(forecasted <= actuals) & (actuals <= 0)].plot(ax=ax, color="orange", marker="o", ls="None")

        # VaR
        forecasted.plot(ax=ax, grid=False, color="black", rot=0)

        # Axes
        plt.xlabel("")
        plt.ylabel("Log Return")
        ax.yaxis.grid()

        sns.despine()
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":

    returns_file = os.path.join(DATA_DIR, 'returns.txt')
    predictions_file = os.path.join(DATA_DIR, 'quantile_predicitons.txt')

    # Test data: the daily log returns of the IBM stock and the 1% VaR forecasts stemming from a variety of risk models.
    Y = pd.read_csv(returns_file, index_col=0, parse_dates=True).iloc[:, 0]
    X = pd.read_csv(predictions_file, index_col=0, parse_dates=True)

    bt = Backtest(actual=Y, forecast=X.loc[:, "eGARCH-fhs"], alpha=0.01)
    bt.plot()