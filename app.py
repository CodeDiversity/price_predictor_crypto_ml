
import requests
import pandas as pd
from prophet import Prophet
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response 

app = Flask(__name__)


@app.route('/predict/<coinId>')
def predict(coinId):
    results = predictCryptoPrice(coinId)
    return jsonify(results)

if __name__ == '__app__':
    app.run()

def predictCryptoPrice(coinId):
    coinId = coinId
    response = requests.get(f"https://api.coingecko.com/api/v3/coins/{coinId}/market_chart?vs_currency=usd&days=1600&interval=daily")
    data = response.json()
    df = pd.DataFrame(data['prices'], columns=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'], unit='ms')
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict(orient='records')
    

