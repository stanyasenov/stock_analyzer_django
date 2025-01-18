import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import os

class StockAnalyzer:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = self.fetch_data()
        self.scaler = StandardScaler()

    def fetch_data(self):
        df = yf.download(self.ticker, start=self.start, end=self.end)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def add_features(self):
        data = self.data.copy()
        data['Year'] = data.index.year
        data['Month'] = data.index.month
        data['Day'] = data.index.day
        data['DayOfWeek'] = data.index.dayofweek
        data['MA7'] = data['Close'].rolling(window=7).mean()
        data['MA30'] = data['Close'].rolling(window=30).mean()
        data['Volatility'] = data['Close'].rolling(window=7).std()
        data['Pct_Change'] = data['Close'].pct_change() * 100
        data['High_Low_Pct'] = (data['High'] - data['Low']) / data['Low'] * 100
        data['Close_Open_Pct'] = (data['Close'] - data['Open']) / data['Open'] * 100
        self.data = data.dropna()

    def create_monthly_heatmap(self, title, filename):
        self.data['MonthlyReturn'] = self.data['Close'].resample('M').last().pct_change() * 100
        monthly_returns = self.data['MonthlyReturn'].resample('M').last()
        monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.month_name()
        heatmap_data = monthly_returns_df.pivot(index='Year', columns='Month', values='Monthly Return')
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        heatmap_data = heatmap_data[months_order]
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap='RdYlGn', annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Monthly Return (%)'})
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.tight_layout()
        plt.savefig(os.path.join('static', filename))
        plt.close()

    def train_and_evaluate_models(self, models):
        X = self.data[['Open', 'High', 'Low', 'Volume', 'MA7', 'MA30', 'Volatility', 'Pct_Change', 'High_Low_Pct', 'Close_Open_Pct']]
        y = self.data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'model': model, 'y_pred': y_pred, 'mse': mse, 'mae': mae, 'r2': r2}
            # Save the prediction plot
            plt.figure(figsize=(14, 7))
            plt.plot(y_test.index, y_test, label='Actual', color='blue', linewidth=2)
            plt.plot(y_test.index, y_pred, label=f'{name} Predicted', color='orange')
            plt.xlabel('Date')
            plt.ylabel('Close Price (USD)')
            plt.title(f'Actual vs Predicted {self.ticker} Close Price ({name})')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join('static', f'{self.ticker.lower()}_price_{name.replace(" ", "_").lower()}.png'))
            plt.close()
        return results
