import pandas as pd
import numpy as np
from typing import Dict, Any, Callable

CREDIT_NUMERIC_COLS = [
"dti", "dti_joint",
"delinq_2yrs",
"mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
"open_acc", "total_acc", "pub_rec", "acc_now_delinq",
"revol_bal", "revol_util", "total_rev_hi_lim",
"tot_coll_amt", "tot_cur_bal", "total_bal_il",
"open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m",
"mths_since_rcnt_il",
"open_rv_12m", "open_rv_24m",
"max_bal_bc",
"all_util",
"inq_last_6mths", "inq_last_12m", "inq_fi",
"collections_12_mths_ex_med",
]

class CreditHistoryEDA:
    def __init__(self, df: pd.DataFrame, target_col: str = "loan_status"):
        """
        Store the full DataFrame and the name of the target column.
        """
        self.df = df
        self.target_col = target_col
        
    def credit_structure_summary(self) -> pd.DataFrame:
        """
        One row per CREDIT_NUMERIC_COLS column with:
        - column
        - dtype
        - n_missing
        - missing_pct
        - mean (if numeric)
        - std (if numeric)
        """
        dataframe_list = []
        for col in CREDIT_NUMERIC_COLS:
            
            columns_name = col
            dtype = self.df[col].dtypes
            n_missing = self.df[col].isna().sum()
            missing_pct = self.df[col].isna().mean()
            if self.df[col].dtype in self.df.select_dtypes(include='number').dtypes.values:
                mean = self.df[col].mean()
                std = self.df[col].std()
            else:
                mean = np.nan
                std = np.nan                
            
            empty_dict = dict()
            list_of_variables = ["column", "dtype", "n_missing", "missing_pct", "mean", "std"]
            list_of_values = [columns_name, dtype, n_missing, missing_pct, mean, std]
            for var, val in zip(list_of_variables, list_of_values):
                empty_dict[var] = val
            dataframe_list.append(empty_dict)
        
        return pd.DataFrame(dataframe_list, columns = ["column","dtype","n_missing","missing_pct","mean","std"])

    def default_rate_by_bucket(self, col: str, bins: int = 4) -> pd.DataFrame:
        """
        For a numeric credit column (e.g., dti, revol_util),
        create `bins` buckets and compute default rate per bucket.

        Return a DataFrame with columns:
        - bucket (interval)
        - n_loans
        - default_rate
        """
        buckets = pd.qcut(self.df[col], q=bins, duplicates='drop')
        grouped_buckets = self.df.groupby(buckets)[self.target_col].agg(n_loans ="count",
                                                            default_rate = "mean")
        return grouped_buckets.reset_index()


    def correlation_with_default(self) -> pd.Series:
        """
        Compute correlation of each numeric credit column with the target
        (assuming loan_status is encoded as 0/1).
        Return a Series indexed by column name.
        """
        list_of_corr = []
        for col in CREDIT_NUMERIC_COLS:
            correlation = self.df[col].corr(self.df[self.target_col])
            list_of_corr.append(correlation)
        series = pd.Series(data=list_of_corr, index=CREDIT_NUMERIC_COLS)
        return series    
            
def credit_history_report(eda: CreditHistoryEDA) -> Dict[str, Any]: 
    # steps: Dict[str, Callable[[], Any]] =
    step_name = {
    "structure_summary": eda.credit_structure_summary,
    "dti_buckets": lambda: eda.default_rate_by_bucket("dti", bins=5),
    "revol_util_buckets": lambda: eda.default_rate_by_bucket("revol_util", bins=5),
    "correlation_with_default": eda.correlation_with_default,
    }  
    credit_history_report = {name: func() for name, func in step_name.items()}
    return credit_history_report


# This is the end