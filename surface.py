import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import plotly.graph_objects as go
import yfinance as yf
import datetime


# Title and presentation
st.set_page_config(
    page_title="Implied Volatility Scatter Plot",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Volatility Surface")

st.write(
    """
    This app calculates and visualizes the implied volatility for options
    based on market data fetched from Yahoo Finance.

    We are using a 3D scatter plot to represent implied volatility across strike prices and expiration dates because Yahoo Finance may have limited data points for certain options. 
    By using a scatter plot, we avoid contaminating the visualization with interpolation and instead display the actual data points available.
    """
)


# Selectig ticker
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")

try:
    stock = yf.Ticker(ticker)
    S0 = stock.history(period="1d")['Close'].iloc[-1]
    st.sidebar.write(f"Latest Stock Price for {ticker}: {S0:.2f}")
    
    expiration_dates = stock.options
    if not expiration_dates:
        st.sidebar.error("No options data available for this ticker.")
        st.stop()
    
except Exception as e:
    st.sidebar.error(f"Could not fetch data for {ticker}. Please check the ticker symbol or try another one.")
    st.stop()


# Selecting options type, rf and n. of dates to plot
option_type = st.sidebar.selectbox("Option Type", ["Calls", "Puts", "Both"])
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.02, min_value=0.0, step=0.01)
exp_date_limit = st.sidebar.number_input("Number of Expiration Dates to Plot", min_value=1, max_value=len(expiration_dates), value=5, step=1)



# Black-Scholes Model Functions
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility_call(S, K, T, r, market_price):
    def objective_function(sigma):
        return (black_scholes_call_price(S, K, T, r, sigma) - market_price) ** 2

    result = minimize(objective_function, x0=0.2, method='L-BFGS-B')
    return result.x[0] if result.success else np.nan

def implied_volatility_put(S, K, T, r, market_price):
    def objective_function(sigma):
        return (black_scholes_put_price(S, K, T, r, sigma) - market_price) ** 2

    result = minimize(objective_function, x0=0.2, method='L-BFGS-B')
    return result.x[0] if result.success else np.nan

# Define lists to store the points for the scatter plot
time_to_expiry = []
strike_prices = []
implied_volatilities = []
option_types = []

# Loop through expiration dates and strikes to calculate implied volatilities
for expiration_date in expiration_dates[:exp_date_limit]:
    T = (datetime.datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.datetime.now()).days
    try:
        
        option_chain = stock.option_chain(expiration_date)
        
        if option_type in ["Calls", "Both"]:
            calls = option_chain.calls
            
            for _, row in calls.iterrows():
                K = row['strike']
                bid = row['bid']
                ask = row['ask']
                
                # Use bid-ask midpoint if available, otherwise use last price
                if bid > 0 and ask > 0:
                    market_price = (bid + ask) / 2
                else:
                    market_price = row['lastPrice']
                
                
                iv = implied_volatility_call(S0, K, T/365.0, r, market_price)
                
                # Append only realistic values to the lists
                if 0.01 < iv < 3.0:
                    time_to_expiry.append(T)
                    strike_prices.append(K)
                    implied_volatilities.append(iv)
                    option_types.append("Call")
        
        if option_type in ["Puts", "Both"]:
            puts = option_chain.puts
            
            for _, row in puts.iterrows():
                K = row['strike']
                bid = row['bid']
                ask = row['ask']
                
                # Use bid-ask midpoint if available, otherwise use last price
                if bid > 0 and ask > 0:
                    market_price = (bid + ask) / 2
                else:
                    market_price = row['lastPrice']
                
                
                iv = implied_volatility_put(S0, K, T/365.0, r, market_price)
                
                # Append only realistic values to the lists
                if 0.01 < iv < 3.0:
                    time_to_expiry.append(T)
                    strike_prices.append(K)
                    implied_volatilities.append(iv)
                    option_types.append("Put")
                    
    except Exception as e:
        continue

# Plot the volatility surface using Plotly
fig = go.Figure()


if "Call" in option_types:
    fig.add_trace(go.Scatter3d(
        x=[strike for strike, opt in zip(strike_prices, option_types) if opt == "Call"],
        y=[expiry for expiry, opt in zip(time_to_expiry, option_types) if opt == "Call"],
        z=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Call"],
        mode='markers',
        marker=dict(
            size=5,
            color=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Call"],
            colorscale='Viridis',
            colorbar=dict(title="IV Calls", x=1.2),
            opacity=0.8
        ),
        name='Call',
    ))


if "Put" in option_types:
    fig.add_trace(go.Scatter3d(
        x=[strike for strike, opt in zip(strike_prices, option_types) if opt == "Put"],
        y=[expiry for expiry, opt in zip(time_to_expiry, option_types) if opt == "Put"],
        z=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Put"],
        mode='markers',
        marker=dict(
            size=5,
            color=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Put"],
            colorscale='Plasma',
            colorbar=dict(title="IV Puts", x=-0.3),  # Positioning the color bar for puts
            opacity=0.8
        ),
        name='Put'
    ))


fig.update_layout(
    title=f"Implied Volatility Scatter Plot for {ticker}",
    scene=dict(
        xaxis=dict(title="Strike Price (K)"),
        yaxis=dict(title="Days to Expiry (T)"),
        zaxis=dict(title="Implied Volatility (σ)")
    ),
    autosize=False,
    width=1000,
    height=700,
)


st.plotly_chart(fig, use_container_width=True)








