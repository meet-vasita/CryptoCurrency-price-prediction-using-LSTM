from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import os

app = Flask(__name__, template_folder="views", static_folder="static")

# Function to predict future prices
def predict_future(model, scaler, last_100, no_of_days=10):
    future_predictions = []
    for i in range(no_of_days):
        next_day = model.predict(last_100).tolist()
        last_100[0].append(next_day[0])
        last_100 = [last_100[0][1:]]
        future_predictions.append(scaler.inverse_transform(next_day))
    return future_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        stock_symbol = request.form['stock_symbol']
        num_days = int(request.form['num_days'])
        
        # Fetch stock data
        end = datetime.now()
        start = datetime(end.year - 6, end.month, end.day)
        stock_data = yf.download(stock_symbol, start, end)
        
        # Load the pre-trained model
        model = load_model("Latest_bit_coin_model.keras")
        
        # Prepare data for prediction
        splitting_len = int(len(stock_data) * 0.9)
        x_test = pd.DataFrame(stock_data.Close[splitting_len:])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']].values)
        x_data = []
        y_data = []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])
        x_data, y_data = np.array(x_data), np.array(y_data)
        
        # Make predictions
        predictions = model.predict(x_data)
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)
        ploting_data = pd.DataFrame({
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        }, index=stock_data.index[splitting_len + 100:])
        
        # Future predictions
        last_100 = stock_data[['Close']].tail(100)
        last_100_scaled = scaler.fit_transform(last_100['Close'].values.reshape(-1, 1)).reshape(1, -1, 1)
        last_100_list = last_100_scaled.tolist()  # Convert to list
        future_results = predict_future(model, scaler, last_100_list, no_of_days=num_days)

        prediction_factor = 0.03  # You can adjust this value to control the level of randomness
        # Create a list to hold the perturbed future predictions
        prediction_future_results = [result * (1 + np.random.uniform(-prediction_factor, prediction_factor)) for result in future_results]

        # Plotting closing prices of the selected cryptocurrency
        closing_fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'Closing Prices of {stock_symbol}')
        closing_fig.update_xaxes(title_text='Time')
        closing_fig.update_yaxes(title_text='Price')
        closing_plot_path = os.path.join('static', f'{stock_symbol.lower()}_closing_plot.html')  # Update plot file name
        closing_fig.write_html(closing_plot_path)
        
        # Plotting actual vs predicted prices with extended future predictions using Plotly Express
        fig = px.line(ploting_data, x=ploting_data.index, y=['original_test_data', 'predictions'], 
                      labels={'value': 'Price', 'variable': 'Type'}, 
                      title=f'Actual vs Predicted Prices of {stock_symbol}')
        fig.add_scatter(x=pd.date_range(ploting_data.index[-1], periods=num_days, freq='D')[1:], 
                        y=np.array(prediction_future_results).reshape(-1), mode='lines', name='Future Predictions', line=dict(color='green'))
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Price')
        plot_path = os.path.join('static', f'{stock_symbol.lower()}_prediction_plot.html')
        fig.write_html(plot_path)
        
        # Plotting stock data as bar graph
        stock_data_bar_fig = px.bar(stock_data.tail(num_days), x=stock_data.tail(num_days).index, y='Close', title=f'{stock_symbol} Stock Data', color='Close', color_continuous_scale='viridis')
        stock_data_bar_path = os.path.join('static', f'{stock_symbol.lower()}_stock_data_bar.html')
        stock_data_bar_fig.write_html(stock_data_bar_path)

        # Plotting original_test_data vs predictions as a grouped bar graph
        original_vs_predictions_bar_fig = px.bar(ploting_data.tail(num_days), x=ploting_data.tail(num_days).index,
                                                 y=['original_test_data', 'predictions'],
                                                 title=f'Original Test Data vs Predictions for {stock_symbol}',
                                                 labels={'value': 'Price', 'variable': 'Type'},
                                                 color_discrete_map={'original_test_data': '#007bff', 'predictions': '#28a745'},
                                                 barmode='group')
        original_vs_predictions_bar_path = os.path.join('static', f'{stock_symbol.lower()}_original_vs_predictions_grouped_bar.html')
        original_vs_predictions_bar_fig.write_html(original_vs_predictions_bar_path)

        return render_template('result.html', predictions=ploting_data.tail(num_days).to_html(classes='table table-striped', index=False), 
                               plot_path=plot_path, closing_plot_path=closing_plot_path, 
                               stock_bar_path=stock_data_bar_path, original_vs_predictions_bar_path=original_vs_predictions_bar_path)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
