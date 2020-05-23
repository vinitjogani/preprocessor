import pandas as pd


class Loader:

    @staticmethod
    def clean_string(s):
        s = s.replace('\n', ' ')
        s = s.replace(':', ' ')
        s = s.replace('   ', ' ')
        s = s.replace('  ', ' ')
        s = s.replace(' ', '_')
        s = s.replace('%', 'pct')
        s = s.replace(' /', '/')
        s = s.replace('/ ', '/')
        s = s.replace('/', '_per_')
        return s.lower()

    @staticmethod
    def rename_columns(df):
        for column in df:
            df.rename(
                columns={column: Loader.clean_string(column)},
                inplace=True
            )

    @staticmethod
    def parse_dates(df):
        for column in df:
            if 'date' in column or 'time' in column:
                try:
                    df[column] = pd.to_datetime(df[column])
                except Exception as e:
                    print(e)
                    continue

    @staticmethod
    def load(path):
        df = pd.read_csv(path)
        Loader.rename_columns(df)
        Loader.parse_dates(df)
        return df
