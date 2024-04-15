import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, BayesianRidge
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


if __name__ == '__main__':
    raw_data_path = os.path.join('.', 'RawData', 'wine+quality')
    files = [item.path for item in os.scandir(raw_data_path) if item.is_file()]

    dataframes = [(file, pd.read_csv(file, delimiter=';')) for file in files if '.csv' in file]

    pd.options.display.max_columns = 12
    pd.options.display.width = 0
    separator = '\n  '
    for filename, df in dataframes:
        print(f'File: {filename}\n{df.describe().round(decimals=2)}')
        na_count = separator.join([f'{x} : {df[x].isnull().sum()}' for x in df.columns if df[x].isnull().sum() > 0])
        print(f'Missing values:\n  {na_count}\n')

    for filename, df in dataframes:
        train_scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(train_scaler.fit_transform(df))
        df_scaled.columns = df.columns
        x_col = 'residual sugar'
        y_col = 'quality'
        hues = [df_scaled.columns[x] for x in range(len(df_scaled.columns)) if (x != 4 and x != len(df_scaled.columns)-1)]
        #hues = df_scaled.columns[:4].values + df_scaled.columns[5:-1].values

        sns.lmplot(data = df_scaled, x=x_col, y=y_col)

        y = df_scaled.pop('quality')
        X = df_scaled
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        linReg = LinearRegression()
        ridgeReg = Ridge()
        SGDReg = SGDRegressor(random_state=42)
        lassoReg = Lasso(alpha=0.01)
        bayesRidgeReg = BayesianRidge()
        linScore = cross_val_score(linReg, X_train, y_train, cv=5)
        ridgeScore = cross_val_score(ridgeReg, X_train, y_train, cv=5)
        SGDScore = cross_val_score(SGDReg, X_train, y_train, cv=5)
        lassoScore = cross_val_score(lassoReg, X_train, y_train, cv=5)
        bayesRidgeScore = cross_val_score(bayesRidgeReg, X_train, y_train, cv=5)
        print(f'Input file: {filename}\n')
        print(f'linear_regressor score: {linScore.mean()} sd: {linScore.std()}')
        print(f'ridge_regressor score: {ridgeScore.mean()} sd: {ridgeScore.std()}')
        print(f'SGD_regressor score: {SGDScore.mean()} sd: {SGDScore.std()}')
        print(f'lasso_regressor score: {lassoScore.mean()} sd: {lassoScore.std()}')
        print(f'BayesianRidge_regressor score: {bayesRidgeScore.mean()} sd: {bayesRidgeScore.std()}')





