import pandas as pd


class Stat:

    NULL_RATE_THRESHOLD = 0.95
    UNIQUE_THRESHOLD = 200

    def __init__(self, df, column):
        self.name = column
        self.null_rate = round(pd.isnull(df[column]).sum() * 100 / df.shape[0], 2)
        self.dtype = df[column].dtype
        self.unique = len(df[column].unique())
        self.unique_rate = round(self.unique / df.shape[0] * 100, 2)

    def exclude(self):
        return (
            self.null_rate > self.NULL_RATE_THRESHOLD * 100
            or self.unique == 1
            or (self.onehot() and self.unique > self.UNIQUE_THRESHOLD)
        )

    def onehot(self):
        return self.dtype == 'object'

    def datefeat(self):
        return str(self.dtype).startswith('date')

    def scale_robust(self):
        return self.dtype == 'float64'

    def preference(self):
        score = 0
        score += self.null_rate
        if self.unique == 1:
            score += 100
        if self.onehot():
            score += 100 * (self.unique > self.UNIQUE_THRESHOLD)
        return score
