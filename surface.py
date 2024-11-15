import streamlit as st
import numpy as np
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import yfinance as yf
import datetime


# Set up Streamlit page
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



# Select ticker
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



# Sidebar inputs
option_type = st.sidebar.selectbox("Option Type", ["Calls", "Puts", "Both"])
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.02, min_value=0.0, step=0.01)
exp_date_limit = st.sidebar.number_input("Number of Expiration Dates to Plot", min_value=1, max_value=len(expiration_dates), value=5, step=1)




# Black-Scholes functions
def call(S, K, T, r, sigma, q=0):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def put(S, K, T, r, sigma, q=0):
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

# Newton-Raphson for implied volatility
def impliedVol_call(p, S, K, T, r, q=0, max_iter=200, tol=1e-4):
    def price_diff(sigma):
        return call(S, K, T, r, sigma, q) - p
    
    def vega(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    sigma_guess = 0.3
    for _ in range(max_iter):
        diff = price_diff(sigma_guess)
        if abs(diff) < tol:
            return sigma_guess
        sigma_guess -= diff / vega(sigma_guess)
    return None

def impliedVol_put(p, S, K, T, r, q=0, max_iter=200, tol=1e-4):
    def price_diff(sigma):
        return put(S, K, T, r, sigma, q) - p
    
    def vega(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    sigma_guess = 0.3
    for _ in range(max_iter):
        diff = price_diff(sigma_guess)
        if abs(diff) < tol:
            return sigma_guess
        sigma_guess -= diff / vega(sigma_guess)
    return None



# Collect data for the scatter plot
time_to_expiry, strike_prices, implied_volatilities, option_types, market_prices = [], [], [], [], []

# Parallelize volatility calculations
def process_option_data(row, S0, T, r, opt_type):
    K = row['strike']
    bid, ask = row['bid'], row['ask']
    if bid > 0 and ask > 0:
        market_price = (bid + ask) / 2
    elif row['lastPrice'] > 0:
        market_price = row['lastPrice']
    else:
        return None

    iv = None
    if opt_type == "Call":
        iv = impliedVol_call(market_price, S0, K, T / 365.0, r)
    elif opt_type == "Put":
        iv = impliedVol_put(market_price, S0, K, T / 365.0, r)

    if iv is not None and 0.01 < iv < 3.0:
        return (T, K, iv, opt_type, market_price)

# Process expiration dates and options
for expiration_date in expiration_dates[:exp_date_limit]:
    expiration_dt = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
    if expiration_dt <= datetime.datetime.now():
        continue

    T = (expiration_dt - datetime.datetime.now()).days
    option_chain = stock.option_chain(expiration_date)

    # Process calls and puts in parallel
    with ThreadPoolExecutor() as executor:
        if option_type in ["Calls", "Both"]:
            calls = option_chain.calls
            results = executor.map(lambda row: process_option_data(row, S0, T, r, "Call"), [row for _, row in calls.iterrows()])
            for result in results:
                if result:
                    T, K, iv, opt, mp = result
                    time_to_expiry.append(T)
                    strike_prices.append(K)
                    implied_volatilities.append(iv)
                    option_types.append(opt)
                    market_prices.append(mp)

        if option_type in ["Puts", "Both"]:
            puts = option_chain.puts
            results = executor.map(lambda row: process_option_data(row, S0, T, r, "Put"), [row for _, row in puts.iterrows()])
            for result in results:
                if result:
                    T, K, iv, opt, mp = result
                    time_to_expiry.append(T)
                    strike_prices.append(K)
                    implied_volatilities.append(iv)
                    option_types.append(opt)
                    market_prices.append(mp)



st.write("Number of points:", len(implied_volatilities))

# Plot only if data is available
if len(implied_volatilities) > 0:

    fig = go.Figure()

    if "Call" in option_types:
        fig.add_trace(go.Scatter3d(
            x=[strike for strike, opt in zip(strike_prices, option_types) if opt == "Call"],
            y=[expiry for expiry, opt in zip(time_to_expiry, option_types) if opt == "Call"],
            z=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Call"],
            customdata=[mp for mp, opt in zip(market_prices, option_types) if opt == "Call"],
            mode='markers',
            marker=dict(
                size=5,
                color=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Call"],
                colorscale='Viridis',
                colorbar=dict(title="IV Calls", x=1.2),
                opacity=0.8),
            hovertemplate=("Strike Price: %{x}<br>Days to Expiry: %{y}<br>IV: %{z:.4f}<br>Market Price: %{customdata:.2f}"),
            name='Call',
        ))

    if "Put" in option_types:
        fig.add_trace(go.Scatter3d(
            x=[strike for strike, opt in zip(strike_prices, option_types) if opt == "Put"],
            y=[expiry for expiry, opt in zip(time_to_expiry, option_types) if opt == "Put"],
            z=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Put"],
            customdata=[mp for mp, opt in zip(market_prices, option_types) if opt == "Put"],
            mode='markers',
            marker=dict(
                size=5,
                color=[iv for iv, opt in zip(implied_volatilities, option_types) if opt == "Put"],
                colorscale='Plasma',
                colorbar=dict(title="IV Puts", x=-0.2),
                opacity=0.8),
                hovertemplate=("Strike Price: %{x}<br>Days to Expiry: %{y}<br>IV: %{z:.4f}<br>Market Price: %{customdata:.2f}"),
            name='Put'
        ))

    fig.update_layout(
        title=f"Implied Volatility Scatter Plot for {ticker}",
        scene=dict(
            xaxis=dict(title="Strike Price (K)"),
            yaxis=dict(title="Days to Expiry (T)", autorange='reversed'),
            zaxis=dict(title="Implied Volatility (σ)")
        ),
        autosize=False,
        width=1200,
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No valid implied volatility data to display.")    
