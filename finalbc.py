
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import tkinter as tk
from tkinter import simpledialog
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions
import requests
from io import StringIO
import time
from requests.exceptions import HTTPError
from datetime import datetime
from time import sleep
import warnings
import plotly.express as px
import plotly.graph_objects as go


def collect_user_data():
    print("Welcome to the Investment Profiling System")
    print("Please answer the following questions to help us understand your investment preferences.")

    while True:
        try:
            capital = float(input("How much capital (in USD) do you wish to invest? "))
            if capital <= 0:
                raise ValueError("Capital must be greater than zero.")
            break
        except ValueError as e:
            print("Invalid input. Please enter a valid number for capital.")

    valid_goals = {1, 2, 3}
    while True:
        try:
            goal_choice = int(input("What are your investment goals? Choose one of the following:\n1: Just experimenting (e.g., learning about investing)\n2: Specific purchase (e.g., buying a car, down payment for a home)\n3: Long-term savings (e.g., retirement)\nEnter the number corresponding to your goal: "))
            if goal_choice not in valid_goals:
                raise ValueError("Invalid goal choice.")
            break
        except ValueError:
            print("Invalid input. Please enter 1, 2, or 3.")

    valid_assets = {'1', '2', '3', '4', '5', '6'}
    while True:
        asset_choices = input("Select the types of assets you are interested in by entering the numbers separated by commas:\n1: Stocks\n2: Bonds\n3: Cryptocurrencies\n4: Commodities\n5: Indices\n6: ETFs\nEnter your choices (e.g., 1, 3, 5): ")
        if all(choice.strip() in valid_assets for choice in asset_choices.split(',')):
            break
        print("Invalid input. Please enter a combination of 1, 2, 3, 4, 5, 6 separated by commas.")

    valid_risks = {'low', 'medium', 'high'}
    while True:
        risk_tolerance = input("What is your risk tolerance? (low, medium, high): ").lower()
        if risk_tolerance in valid_risks:
            break
        print("Invalid input. Please enter 'low', 'medium', or 'high'.")

    while True:
        try:
            duration = int(input("Preferred investment duration (in years): "))
            if duration <= 0:
                raise ValueError("Duration must be greater than zero.")
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer for duration.")

    goal_mapping = {1: 'experimenting', 2: 'specific_purchase', 3: 'long_term_savings'}
    goals = goal_mapping[goal_choice]

    top_assets = {
        'stocks': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'V', 'JNJ', 'WMT'],
        'bonds': ['^TNX', '^FVX', '^IRX'],
        'cryptocurrencies': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'LTC-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD'],
        'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'ALI=F', 'ZN=F'],
        'indices': ['^GSPC', '^DJI', '^IXIC'],
        'ETFs': ['SPY', 'IVV', 'VTI']
    }

    asset_category_map = {'1': 'stocks', '2': 'bonds', '3': 'cryptocurrencies', '4': 'commodities', '5': 'indices', '6': 'ETFs'}

    asset_categories = [asset_category_map[choice.strip()] for choice in asset_choices.split(',') if choice.strip() in asset_category_map]

    selected_top_assets = {category: top_assets[category] for category in asset_categories}


    return {
        'capital': capital,
        'goals': goals,
        'risk_tolerance': risk_tolerance,
        'duration': duration,
        'asset_preferences': selected_top_assets
    }
    
    
def determine_risk_profile(user_data):
    risk_tolerance = user_data['risk_tolerance'].lower()
    asset_preferences = user_data['asset_preferences']

    # Safety hierarchy
    safety_hierarchy = {
        'bonds': 1,
        'indices': 2,
        'ETFs': 3,
        'stocks': 4,
        'commodities': 5,
        'cryptocurrencies': 6
    }

    sorted_preferences = sorted(asset_preferences, key=lambda x: safety_hierarchy[x])


    allocation = {asset: 0 for asset in asset_preferences}
    
    # Determine allocation based on risk tolerance
    if risk_tolerance == 'high':
        # assign more to the least safe assets
        allocation[sorted_preferences[-1]] = 70  # most allocation to the boldest asset
        remaining_allocation = (100 - 70) / (len(sorted_preferences) - 1)
        for asset in sorted_preferences[:-1]:
            allocation[asset] = remaining_allocation
    elif risk_tolerance == 'low':
        # assign more to the safest assets
        allocation[sorted_preferences[0]] = 70  # most allocation to the safest asset
        remaining_allocation = (100 - 70) / (len(sorted_preferences) - 1)
        for asset in sorted_preferences[1:]:
            allocation[asset] = remaining_allocation
    else:
        # moderate risk tolerance: distribute evenly
        even_allocation = 100 / len(sorted_preferences)
        for asset in sorted_preferences:
            allocation[asset] = even_allocation

    return allocation


def distribute_within_categories(capital, allocation, top_assets):
    detailed_allocation = {}
    for asset_category, percentage in allocation.items():
        total_category_capital = (percentage / 100) * capital
        num_assets = len(top_assets[asset_category])
        per_asset_allocation = total_category_capital / num_assets
        detailed_allocation[asset_category] = {asset: per_asset_allocation for asset in top_assets[asset_category]}
    return detailed_allocation



def fetch_historical_data6(selected_assets, start_date="2020-01-01", end_date="2024-01-01", retries=3):
    data = {}
    errors = {}

    for category, assets in selected_assets.items():
        for asset in assets:
            attempt = 0
            success = False
            while attempt < retries and not success:
                try:
                    print(f"Attempting to download data for {asset}... (Attempt {attempt + 1})")
                    df = yf.download(asset, start=start_date, end=end_date, progress=False)
                    if not df.empty:
                        cleaned_data = df['Adj Close'].replace(0, pd.NA).dropna()
                        if cleaned_data.empty:
                            raise ValueError("Data after cleaning is empty")
                        data[asset] = cleaned_data
                        success = True
                    else:
                        raise ValueError("Downloaded data frame is empty")
                except Exception as e:
                    print(f"Failed to fetch data for {asset} on attempt {attempt + 1}: {e}")
                    time.sleep(2)  # Sleep before the next retry so I dont overwhelm API
                    attempt += 1
                    if attempt == retries:
                        errors[asset] = str(e)

    if errors:
        print("Failed downloads:")
        for asset, error in errors.items():
            print(f"[{asset}]: {error}")

    return data





def calculate_asset_performance(asset_data_dict):
    """
    Calculates performance metrics for given asset data.
    Returns:
        dict: A dictionary containing performance metrics (annual returns, annual volatility, correlation matrix) for each asset.
    """
    performance_metrics = {}
    
    # Combining all assets data for correlation matrix calculation
    all_daily_returns = pd.DataFrame()

    for asset, data in asset_data_dict.items():
        if data is not None and not data.empty:
            daily_returns = data.pct_change()
            daily_returns.replace([np.inf, -np.inf], pd.NA, inplace=True)  # Handle infinite returns
            daily_returns.dropna(inplace=True)  # Drop rows with NaN values
            
            all_daily_returns[asset] = daily_returns.squeeze()
            
            annual_returns = daily_returns.mean() * 252
            annual_volatility = daily_returns.std() * np.sqrt(252)

            performance_metrics[asset] = {
                'annual_returns': annual_returns,
                'annual_volatility': annual_volatility
            }
    
    correlation_matrix = all_daily_returns.corr()
    performance_metrics['correlation_matrix'] = correlation_matrix

    return performance_metrics





def portfolio_optimization(asset_data_dict):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  
        all_prices = pd.concat([data for data in asset_data_dict.values() if data is not None], axis=1)
        mu = mean_historical_return(all_prices)
        S = CovarianceShrinkage(all_prices).ledoit_wolf().astype(float)
        S = (S + S.T) / 2 

        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        performance = ef.portfolio_performance(verbose=True)

        return {'weights': cleaned_weights, 'performance': performance}


def forecast_future_prices(asset_data_dict, forecast_horizon=3):

    forecasted_prices_dict = {}

    for asset, data in asset_data_dict.items():
        if data is not None and not data.empty:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            model = LinearRegression()
            model.fit(X, y) 

            future_indices = np.arange(len(data), len(data) + forecast_horizon * 252).reshape(-1, 1)
            forecasted_data = pd.DataFrame(
                model.predict(future_indices), 
                index=pd.date_range(start=data.index[-1], periods=forecast_horizon * 252, freq='B'),
                columns=[asset]
            )

            forecasted_prices_dict[asset] = forecasted_data

    return forecasted_prices_dict





def plot_detailed_allocation_table(detailed_allocation, total_capital):
    assets = []
    percentages = []
    amounts = []

    for category, assets_info in detailed_allocation.items():
        for asset, amount in assets_info.items():
            assets.append(asset)
            percentage = (amount / total_capital) * 100
            percentages.append(f"{percentage:.2f}%")
            amounts.append(f"${amount:,.2f}")


    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.set_title('Detailed Capital Allocation')


    table_data = list(zip(assets, percentages, amounts))

    table = ax.table(cellText=table_data, colLabels=['Asset', 'Percentage of Total Capital', 'Amount Allocated'], cellLoc = 'center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  

    plt.show()


def plot_performance_table(performance_metrics):
    """
    Plots a table of performance metrics for each asset.

    Parameters:
    - performance_metrics (dict): Dictionary containing performance metrics for assets.
    """
    assets = list(performance_metrics.keys())
    annual_returns = [f"{performance_metrics[asset]['annual_returns'] * 100:.2f}%" for asset in assets if 'annual_returns' in performance_metrics[asset]]
    annual_volatility = [f"{performance_metrics[asset]['annual_volatility'] * 100:.2f}%" for asset in assets if 'annual_volatility' in performance_metrics[asset]]

    if 'correlation_matrix' in assets:
        assets.remove('correlation_matrix')

    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.set_title('Asset Performance Metrics')

    # Table data
    table_data = []
    for asset in assets:
        table_data.append([
            asset,
            performance_metrics[asset]['annual_returns'] * 100,  
            performance_metrics[asset]['annual_volatility'] * 100  
        ])

    # Create table
    table = ax.table(cellText=table_data, colLabels=['Asset', 'Annual Returns (%)', 'Annual Volatility (%)'], cellLoc = 'center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  

    plt.show()


def plot_cumulative_returns(historical_data):
    for asset, data in historical_data.items():
        if isinstance(data, pd.Series):
            cumulative_returns = (data / data.iloc[0] - 1) * 100
            plt.figure(figsize=(10, 6))
            plt.plot(cumulative_returns, label=f'{asset} Cumulative Returns')
            plt.title('Cumulative Returns Over Time')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns (%)')
            plt.legend()
            plt.grid(True)
            plt.show()

def plot_volatility(historical_data):
    """Plot the rolling volatility of the portfolio."""
    plt.figure(figsize=(10, 6))

    for asset, data in historical_data.items():
        if isinstance(data, pd.Series):
            rolling_volatility = data.pct_change().rolling(window=252).std() * np.sqrt(252)
            plt.plot(rolling_volatility, label=f'Volatility - {asset}')

    plt.title('Annualized Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_heatmap(historical_data):
    combined_data = pd.DataFrame()
    
    for asset, data in historical_data.items():
        if isinstance(data, pd.Series):
            combined_data[asset] = data.pct_change()

    combined_data.dropna(inplace=True)

    correlations = combined_data.corr() 
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def plot_forecast_vs_actual(historical_data, forecasted_data):
    plt.figure(figsize=(14, 7))
    for asset, hist_data in historical_data.items():
        plt.plot(hist_data.index, hist_data, label=f'Actual - {asset}')
        if asset in forecasted_data:
            plt.plot(forecasted_data[asset].index, forecasted_data[asset], linestyle='--', label=f'Forecast - {asset}')
    plt.title('Forecast vs. Actual Performance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_detailed_allocation(detailed_allocation):
    labels = []
    sizes = []

    for category, assets in detailed_allocation.items():
        for asset, amount in assets.items():
            labels.append(f"{asset} ({category})")
            sizes.append(amount)

    if not sizes:
        print("No allocation data to plot.")
        return

    total = sum(sizes)
    sizes = [s / total * 100 for s in sizes] 

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Detailed Asset Allocation')
    plt.show()


def plot_category_allocation(allocation):
  #pie chart
    if not allocation:
        print("No allocation data to plot.")
        return

    #prepare labels and sizes from the allocation dictionary
    labels = [f'{label} ({size}%)' for label, size in allocation.items()]
    sizes = list(allocation.values())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Category Percentage Allocation')
    plt.show()






#additional data cleaning before I use it in main
def clean_data(data):
    return data.replace([np.inf, -np.inf], np.nan).dropna()






def main():
    #collect user input
    user_data = collect_user_data()
    print(user_data)
    
    #determine asset allocation based on risk profile
    allocation = determine_risk_profile(user_data)
    
    detailed_allocation = distribute_within_categories(user_data['capital'], allocation, user_data['asset_preferences'])

    
    # fetch historical data 
    historical_data = fetch_historical_data6(user_data['asset_preferences'])
    
    cleaned_historical_data = {k: clean_data(v) for k, v in historical_data.items() if isinstance(v, pd.Series)}
    
    
    performance_metrics = calculate_asset_performance(cleaned_historical_data)
    

    
    #Optimize the portfolio
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        optimized_results = portfolio_optimization(cleaned_historical_data)
    #this one somehow prints expected annual return, annual volatility and sharpe ratio withou being called to do so.
        
        
    
    #forecast future prices
    forecasted_prices = forecast_future_prices(cleaned_historical_data)
    
    
    
    
    #visualizations
    plot_category_allocation(allocation)#works
    plot_detailed_allocation(detailed_allocation)#works
    plot_detailed_allocation_table(detailed_allocation, user_data['capital'])#works
    plot_performance_table(performance_metrics)#works
    plot_cumulative_returns(cleaned_historical_data) #works
    plot_volatility(cleaned_historical_data) #works
    plot_correlation_heatmap(cleaned_historical_data)#works
    plot_forecast_vs_actual(cleaned_historical_data, forecasted_prices)#works
    


if __name__ == "__main__":
    main()
