import pandas as pd
from typing import Dict, Any, List, Callable

BORROWER_COLS = [
    "id", "member_id",
    "emp_title", "emp_length",
    "home_ownership",
    "annual_inc", "annual_inc_joint",
    "verification_status", "verification_status_joint",
    "zip_code", "addr_state",
    "purpose", "title", "desc",
    "issue_d", "pymnt_plan", "policy_code",
    "url",
]

class BorrowerProfileEDA:
    def __init__(self, df: pd.DataFrame, target_col: str = "loan_status"):
        """
        Store the full DataFrame and the name of the target column.
        """
        self.df = df
        self.target_col = target_col

    def structure_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per column in BORROWER_COLS:
        - column: column name
        - dtype: pandas dtype
        - n_missing: number of missing values
        - missing_pct: percentage of missing values
        - n_unique: number of unique values
        """
        dataframe_list = []
        for col in BORROWER_COLS:
            
            empty_dict = dict()
            columns_name = col
            dtype = self.df[col].dtypes
            n_missing = self.df[col].isna().sum()
            missing_pct = self.df[col].isna().mean()
            n_unique = self.df[col].nunique(dropna=True)
            
            empty_dict = dict()
            list_of_variables = ["column", "dtype", "n_missing", "missing_pct", "n_unique"]
            list_of_values = [columns_name, dtype, n_missing, missing_pct, n_unique]
            for var, val in zip(list_of_variables, list_of_values):
                empty_dict[var] = val
            dataframe_list.append(empty_dict)
        
        return pd.DataFrame(dataframe_list, columns = ["column", "dtype", "n_missing", "missing_pct", "n_unique"])
            
            
            

    def income_summary(self) -> pd.DataFrame:
        """
        Return basic stats (count, mean, std, min, max, quartiles)
        for:
        - annual_inc
        - annual_inc_joint

        Use df[["annual_inc", "annual_inc_joint"]].describe().T
        or equivalent.
        """
        return self.df[["annual_inc", "annual_inc_joint"]].describe().T

    def categorical_freqs(self, max_levels: int = 10) -> Dict[str, pd.Series]:
        """
        For important categorical borrower columns (e.g. home_ownership,
        addr_state, purpose), return a dict:

            {
              "home_ownership": Series of top levels,
              "addr_state": Series of top levels,
              ...
            }

        Each Series should be the result of value_counts().head(max_levels).
        """
        list_of_dict_variables = ["home_ownership", "addr_state", "purpose"]
        
        info_dict = dict()
        for var in list_of_dict_variables:
            info_dict[var] = self.df[var].value_counts().head(max_levels)
        
        return info_dict
        
        
    def default_rate_by_category(self, col: str) -> pd.Series:
        """
        For a given categorical column (e.g. 'home_ownership' or 'purpose'),
        compute the default rate per category.

        Default rate = mean of self.target_col for each category.
        Return a pandas Series indexed by category, with values in [0, 1].
        """
        return self.df.groupby(col)[self.target_col].mean()
    
    
    
def borrower_eda_steps(eda: BorrowerProfileEDA) -> Dict[str, Callable[[], Any]]:
    """
    Return a dict mapping step names to zero-argument callables.

    Each callable should run ONE piece of EDA when called.
    Example keys (you can adjust names if you like):
        - "structure"
        - "income"
        - "freqs"
        - "default_by_home_ownership"
        - "default_by_purpose"
    """
    return {"structure": eda.structure_summary, "income": eda.income_summary, 
            "freqs": lambda: eda.categorical_freqs(max_levels = 10),
            "default_by_home_ownership": lambda: eda.default_rate_by_category("home_ownership"),
            "default_by_purpose": lambda: eda.default_rate_by_category("purpose")}


def run_borrower_eda_pipeline(eda: BorrowerProfileEDA) -> Dict[str, Any]:
    """
    Run all steps from borrower_eda_steps(eda).

    - Get the dict: step_name -> function
    - Iterate over items
    - Call each function
    - Collect the outputs in a dict: step_name -> result

    This function should clearly show functional programming:
    we store functions in a dict, then loop and call them.
    """
    functions_dict = dict()
    for k,v in borrower_eda_steps(eda).items():
        functions_dict[k] = v()
    return functions_dict