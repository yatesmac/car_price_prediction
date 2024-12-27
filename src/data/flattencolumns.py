from sklearn.base import BaseEstimator, TransformerMixin


class FlattenColumns(BaseEstimator, TransformerMixin):
    def __init__(self, multi_label):
        super().__init__()
        self.multi_label = multi_label

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert each row of the DataFrame into a dictionary;
        # then expand any list columns into multiple {key: 1} entries.
        rows = X.to_dict(orient='records')
        records = []
        for row in rows:
            new_row = {}
            for col, val in row.items():
                if col in self.multi_label and isinstance(val, list):
                    # For list columns, create new keys col=item
                    for item in val:
                        new_row[f"{col}={item}"] = 1
                else:
                    # For normal columns, just keep the original key/value
                    new_row[col] = val
            records.append(new_row)
        
        return records