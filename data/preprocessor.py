from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
import json
import pandas as pd


class Preprocessor:

    def __init__(self, dropna, exclude, onehot, minmax, robust):
        self.exclude = exclude
        self.onehot = list(set(onehot) - set(exclude))
        self.minmax = list(set(minmax) - set(exclude))
        self.robust = list(set(robust) - set(exclude))
        self.dropna = list(set(dropna) - set(exclude))

        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()

    def drop_nulls(self, df):
        remaining = df[self.dropna].dropna().index
        df = df.loc[remaining]
        return df

    def fit_onehot_encoder(self, df):
        filled = df[self.onehot].fillna('%null%')
        self.onehot_encoder.fit(filled)

    def transform_onehot_encoder(self, df):
        filled = df[self.onehot].fillna('%null%')
        matrix = self.onehot_encoder.transform(filled).todense()

        feature_names = self.onehot_encoder.get_feature_names(self.onehot)
        non_null_features = [f for f in feature_names if not f.endswith('%null%')]
        new_df = pd.DataFrame(matrix, columns=feature_names, index=filled.index)[non_null_features]

        others = list(set(df.columns) - set(self.onehot))
        df = pd.concat([df[others], new_df], axis=1)
        return df

    def fit_scalers(self, df):
        if len(self.minmax):
            self.minmax_scaler.fit(df[self.minmax])
        if len(self.robust):
            self.robust_scaler.fit(df[self.robust])

    def transform_scalers(self, df):
        others = list(set(df.columns) - set(self.minmax + self.robust))
        dfs = [df[others]]

        if len(self.minmax):
            data = self.minmax_scaler.transform(df[self.minmax])
            dfs.append(pd.DataFrame(data, columns=self.minmax, index=df.index))

        if len(self.robust):
            data = self.robust_scaler.transform(df[self.robust])
            dfs.append(pd.DataFrame(data, columns=self.robust, index=df.index))

        df = pd.concat(dfs, axis=1)
        return df

    def fit(self, X):
        self.fit_scalers(X)
        self.fit_onehot_encoder(X)

    def transform(self, X):
        X.index = range(X.shape[0])
        include = list(set(X.columns) - set(self.exclude))
        X = X[include]
        X = self.drop_nulls(X)

        X = self.transform_onehot_encoder(X)
        X = self.transform_scalers(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def build(params='params.json'):
        return Preprocessor(**json.load(open(params, 'r')))
