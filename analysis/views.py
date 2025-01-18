from django.shortcuts import render
from .models import StockAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def home(request):

    return render(request, 'home.html')


def analyze(request):

    btc_analyzer = StockAnalyzer("BTC-USD", start="2015-01-01", end="2025-01-01")
    sp500_analyzer = StockAnalyzer("^GSPC", start="2015-01-01", end="2025-01-01")

    btc_analyzer.add_features()
    sp500_analyzer.add_features()

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Linear Regression': LinearRegression(),
    }

    btc_results = btc_analyzer.train_and_evaluate_models(models)

    sp500_results = sp500_analyzer.train_and_evaluate_models({'Linear Regression': LinearRegression()})

    btc_analyzer.create_monthly_heatmap('Bitcoin Monthly Returns (%)', 'bitcoin_monthly_heatmap.png')

    result_data = {
        'btc_results': btc_results,
        'sp500_results': sp500_results,
        'btc_monthly_heatmap': 'bitcoin_monthly_heatmap.png',
        'btc_price_rf': 'btc-price-random_forest.png',
        'btc_price_lr': 'btc-price-linear_regression.png',
        'sp500_price_lr': 'sp500-price-linear_regression.png',
    }

    return render(request, 'analyze.html', result_data)
