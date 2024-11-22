import pandas as pd

class Saver:
    def __init__(self) -> None:
        self.df = pd.DataFrame()

    def add(self, row):
        # self.df = self.df.append(row, ignore_index=True)
        self.df = pd.concat([self.df, pd.DataFrame([row], columns=row.keys())], ignore_index=True)

    def write(self, path):
        self.df.to_csv(path)