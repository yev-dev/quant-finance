import pandas as pd
import numpy as np
import inspect
import time
import warnings

import re
from datetime import datetime, timedelta


def handle_errors(func):
    """
    Decorator to handle specific errors that may occur in a function and provide informative messages.

    Args:
        func (function): The function to be decorated.

    Returns:
        function: The decorated function.

    Raises:
        KeyError: If an index name is missing in the provided financial statements.
        ValueError: If an error occurs while running the function, typically due to incomplete financial statements.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            function_name = func.__name__
            print(
                "There is an index name missing in the provided financial statements. "
                f"This is {e}. This is required for the function ({function_name}) "
                "to run. Please fill this column to be able to calculate the ratios."
            )
            return pd.Series(dtype="object")
        except ValueError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e}"
            )
            return pd.Series(dtype="object")
        except AttributeError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e}"
            )
            return pd.Series(dtype="object")
        except ZeroDivisionError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e} This is due to a division by zero."
            )
            return pd.Series(dtype="object")
        except IndexError as e:
            function_name = func.__name__
            print(
                f"An error occurred while trying to run the function "
                f"{function_name}. {e} This is due to missing data."
            )
            return pd.Series(dtype="object")

    # These steps are there to ensure the docstring of the function remains intact
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__signature__ = inspect.signature(func)
    wrapper.__module__ = func.__module__

    return wrapper


class Strategy(object):

    def __init__(self) -> None:
        pass

class Model(object):

    def __init__(self, start_date: str | None = None,
        end_date: str | None = None, df: pd.DataFrame | pd.Series = None, strategy: Strategy = None, *args, **kwargs) -> None:

        if start_date and re.match(r"^\d{4}-\d{2}-\d{2}$", start_date) is None:
            raise ValueError(
                "Please input a valid start date (%Y-%m-%d) like '2010-01-01'"
            )
        if end_date and re.match(r"^\d{4}-\d{2}-\d{2}$", end_date) is None:
            raise ValueError(
                "Please input a valid end date (%Y-%m-%d) like '2020-01-01'"
            )
        if start_date and end_date and start_date > end_date:
            raise ValueError(
                f"Please ensure the start date {start_date} is before the end date {end_date}"
            )
        
        self.verbose = kwargs.pop("verbose", False)

        if isinstance(df, pd.DataFrame):
            self.df = df

            if self.df.name is not None and self.df.name != "":
                df_name = str(self.df.name)
            else:
                df_name = "DataFrame"

            if self.verbose: print(f"[i] Loaded {df_name}{self.df.shape}")

    
    def _validate_ta_strategy(self, strategy):
        if strategy is not None or isinstance(strategy, Strategy):
            self.strategy = strategy
        elif len(self.strategy_ta) > 0:
            print(f"[+] Strategy: {self.strategy_name}")
        else:
            pass  

    def _validate_additional_kwargs(self, **kwargs):
        """Chart Settings"""

        default_mpf_width = {
            'candle_linewidth': 0.6,
            
        }

        self.config = {}

        self.config["rpad"] = kwargs.pop("rpad", 10)
        self.config["title"] = kwargs.pop("title", "Asset")
        self.config["volume"] = kwargs.pop("volume", True)

        self.config["width_config"] = kwargs.pop("width_config", default_mpf_width)



    @handle_errors
    def get_conditional_value_at_risk(
        self,
        period: str | None = None,
        alpha: float = 0.05,
        within_period: bool = True,
        rounding: int | None = None,
        growth: bool = False,
        lag: int | list[int] = 1,
        distribution: str = "historic",
    ):
        pass

    def simulate_data(self):

        import random


        random_seeds = {
            "Stock A": random.seed(1),
            "Stock B": random.seed(2),
            "Stock C": random.seed(3),
            "Stock D": random.seed(4),
            "Stock E": random.seed(5),
            "Stock F": random.seed(6),
        }

        return_statistics = {}


        for stock_name, random_seed in random_seeds.items():
            returns = pd.Series([random.randint(-15, 15) / 100 for _ in range(1, 100)])

            return_statistics[stock_name] = {}

            return_statistics[stock_name]["Standard Deviation"] = returns.std()

            return_statistics[stock_name]["Skewness"] = risk_model.get_skewness(returns=returns)

            return_statistics[stock_name]["Kurtosis"] = risk_model.get_kurtosis(returns=returns)

            return_statistics[stock_name]["Max Drawdown"] = risk_model.get_max_drawdown(
                returns=returns
            )

            return_statistics[stock_name]["Value at Risk"] = var_model.get_var_historic(
                returns=returns, alpha=0.05
            )

            return_statistics[stock_name]["Expected Shortfall"] = evar_model.get_evar_gaussian(
                returns=returns, alpha=0.05
            )

        pd.DataFrame(return_statistics)


MULTI_PERIOD_INDEX_LEVELS = 2

def get_var_historic(
    returns: pd.Series | pd.DataFrame, alpha: float
) -> pd.Series | pd.DataFrame:
    """
    Calculate the historical Value at Risk (VaR) of returns.

    Args:
        returns (pd.Series | pd.DataFrame): A Series or Dataframe of returns.
        alpha (float): The confidence level (e.g., 0.05 for 95% confidence).

    Returns:
        pd.Series | pd.DataFrame: VaR values as float if returns is a pd.Series,
        otherwise as pd.Series or pd.DataFrame with time as index.
    """
    if isinstance(returns, pd.DataFrame):
        if returns.index.nlevels == MULTI_PERIOD_INDEX_LEVELS:
            periods = returns.index.get_level_values(0).unique()
            period_data_list = []

            for sub_period in periods:
                period_data = returns.loc[sub_period].aggregate(
                    get_var_historic, alpha=alpha
                )
                period_data.name = sub_period

                if not period_data.empty:
                    period_data_list.append(period_data)

            value_at_risk = pd.concat(period_data_list, axis=1)

            return value_at_risk.T

        return returns.aggregate(get_var_historic, alpha=alpha)
    if isinstance(returns, pd.Series):
        return np.percentile(
            returns, alpha * 100
        )  # The actual calculation without data wrangling

    raise TypeError("Expects pd.DataFrame or pd.Series, no other value.")


