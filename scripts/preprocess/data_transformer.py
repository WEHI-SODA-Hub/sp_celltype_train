from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
import pandas as pd
import numpy as np
import sys
import json
from typing import Dict


class DataTransformer:
    def __init__(self) -> None:
        self.title = "Data Transformer"
        self.valid_scheme = {
            "logp1": self.logp1_transform,
            "poly": self.polyfeatures_transform,
            "percentile": self.percentile_transform,
        }

    def transform_data(
        self, X: pd.DataFrame, transform_scheme: str, args: Dict = None
    ) -> pd.DataFrame:
        """
        Apply the correct transform based on the given string
        """
        if transform_scheme not in self.valid_scheme:
            print(
                "{}: Valid scheme for data transformation not found. Returning original X.".format(
                    self.title
                )
            )
            return X
        return self.valid_scheme[transform_scheme](X=X, args=args)

    def logp1_transform(self, X: pd.DataFrame, args=None) -> pd.DataFrame:
        """
        log(X+1) transform applied to all columns
        """
        print()
        df = pd.DataFrame(FunctionTransformer(np.log1p).fit_transform(X))
        df.columns = "Logp1 Transformer: " + X.columns
        return df

    def percentile_transform(self, X: pd.DataFrame, args=None) -> pd.DataFrame:
        """
        Percentile normalization: clips values to specified percentiles and scales to [0, 1].
        Default: 1st percentile -> 0, 99th percentile -> 1
        
        Args:
            X: Input dataframe
            args: Optional dict with 'lower_percentile' (default 1) and 'upper_percentile' (default 99)
        """
        lower_percentile = args.get("lower_percentile", 1) if args else 1
        upper_percentile = args.get("upper_percentile", 99) if args else 99
        
        print(f"INFO: Applying percentile normalization (p{lower_percentile} -> 0, p{upper_percentile} -> 1)")
        
        df = X.copy()
        
        # Calculate percentiles for each column
        lower_bounds = df.quantile(lower_percentile / 100.0, axis=0)
        upper_bounds = df.quantile(upper_percentile / 100.0, axis=0)
        
        # Clip values to percentile range
        df = df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
        
        # Scale to [0, 1] range
        # Handle case where lower == upper (constant column)
        range_vals = upper_bounds - lower_bounds
        range_vals = range_vals.replace(0, 1)  # Avoid division by zero
        
        df = (df - lower_bounds) / range_vals
        
        df.columns = f"Percentile (p{lower_percentile}-p{upper_percentile}): " + X.columns
        
        return df

    def polyfeatures_transform(self, X: pd.DataFrame, args: Dict) -> pd.DataFrame:
        """
        Polynomial features
        """
        poly = PolynomialFeatures(
            degree=args["degree"], interaction_only=args["interaction_only"]
        )
        combinations_options = args["combinations"]
        x_list = [X]
        for key, val in combinations_options.items():
            print("INFO: Determining polynomial features for {}".format(key))
            cols_of_interest = [x for x in X.columns if any(y in x for y in val)]
            X_filtered = X.loc[:, cols_of_interest]
            X_transformed_arr = poly.fit_transform(X_filtered)
            feature_names = poly.get_feature_names_out(input_features=cols_of_interest)

            X_transformed = pd.DataFrame(X_transformed_arr, columns=feature_names)
            X_transformed = X_transformed.drop(cols_of_interest, axis=1)
            x_list.append(X_transformed)

        df = pd.concat(x_list, axis=1)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df
