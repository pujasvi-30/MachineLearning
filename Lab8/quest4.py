#one-hot encoding
import numpy as np
def one_hot_numpy(column):
    unique_values = np.unique(column)

    one_hot = np.zeros((len(column), len(unique_values)))

    for i, value in enumerate(unique_values):
        one_hot[:, i] = (column == value).astype(int)

    return one_hot, unique_values


# def one_hot_dataframe(df, categorical_columns):
#     df_copy = df.copy()
#
#     for col in categorical_columns:
#         encoded = one_hot_encode(df_copy[col])
#         df_copy = df_copy.drop(columns=[col])
#         df_copy = pd.concat([df_copy, encoded], axis=1)
#
#     return df_copy

# ordinal encoding

import pandas as pd


def ordinal_encode(column):
    unique_values = sorted(column.unique())

    # Create mapping dictionary
    mapping = {value: index for index, value in enumerate(unique_values)}

    # Replace values using mapping
    encoded_column = column.map(mapping)

    return encoded_column, mapping