from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
import json
import pandas as pd


class Preprocessor:

    def __init__(self, dropna, exclude, onehot, minmax, robust, datefeats, copy):
        self.exclude = exclude
        self.onehot = list(set(onehot) - set(exclude))
        self.minmax = list(set(minmax) - set(exclude))
        self.robust = list(set(robust) - set(exclude))
        self.dropna = list(set(dropna) - set(exclude))
        self.datefeats = list(set(datefeats) - set(exclude))
        self.copy = list(set(copy) - set(exclude))

        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = QuantileTransformer(output_distribution='normal')

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
        if len(self.minmax):
            df[self.minmax] = self.minmax_scaler.transform(df[self.minmax])
        if len(self.robust):
            df[self.robust] = self.robust_scaler.transform(df[self.robust])
        return df

    def fit_dates(self, df):
        for feat in self.datefeats:
            self.minmax.extend([
                feat + "_day", feat + "_month",
                feat + "_year", feat + "_dayofweek",
            ])
        self.minmax = list(set(self.minmax))

    def transform_dates(self, df):
        for feat in self.datefeats:
            df[feat + "_day"] = df[feat].map(lambda x: x.day)
            df[feat + "_month"] = df[feat].map(lambda x: x.month)
            df[feat + "_year"] = df[feat].map(lambda x: x.year)
            df[feat + "_dayofweek"] = df[feat].map(lambda x: x.dayofweek)
        return df[set(df.columns) - set(self.datefeats)]

    def copy_features(self, df):
        for feat in self.copy:
            df[feat + "_copy"] = df[feat]
        return df

    def fit(self, X):
        self.fit_dates(X)
        self.transform_dates(X)

        self.fit_scalers(X)
        self.fit_onehot_encoder(X)

    def transform(self, X):
        X.index = range(X.shape[0])
        include = list(set(X.columns) - set(self.exclude))
        X = X[include]

        X = self.drop_nulls(X)
        X = self.copy_features(X)
        X = self.transform_dates(X)
        X = self.transform_onehot_encoder(X)
        X = self.transform_scalers(X)

        X.fillna(0, inplace=True)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def build(params='params.json'):
        return Preprocessor(**json.load(open(params, 'r')))
