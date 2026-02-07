import numpy as np
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timezone

# ======================
# PARAMETERS
# ======================

TICKER = "NOC"
RISK_FREE_RATE = 0.04
ITM_FACTOR = 0.6 #Factor for excluding deep In-The-Money Options
OTM_FACTOR = 1.25 #Factor for excluding deep Out-The-Money Options
SMOOTH_SIGMA = 1.5
FORWARD_TOL = 0.05
MIN_DAYS = 30
MAX_DAYS = 365

# ======================
# BLACK–SCHOLES
# ======================

def bs_call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ======================
# SSVI TOTAL VARIANCE
# ======================

def ssvi_total_variance(k, theta, rho, phi):
    return 0.5 * theta * (
        1 + rho * phi * k +
        np.sqrt((phi * k + rho)**2 + 1 - rho**2)
    )

# ======================
# FIT SSVI FOR ONE EXPIRY
# ======================

def fit_ssvi(calls, puts, spot, T):
    F = spot * np.exp(RISK_FREE_RATE * T)

    strikes = []
    ivs = []

    for df in (calls, puts):
        for _, row in df.iterrows():
            K = row["strike"]

            if K < ITM_FACTOR * spot or K > OTM_FACTOR * spot:
                continue

            iv = row["impliedVolatility"]
            if iv <= 0 or np.isnan(iv):
                continue

            strikes.append(K)
            ivs.append(iv)

    strikes = np.array(strikes)
    ivs = np.array(ivs)

    if len(strikes) < 10:
        return None

    k = np.log(strikes / F)
    w = ivs**2 * T

    theta0 = np.mean(w)
    rho0 = -0.3
    phi0 = 0.2

    bounds = (
        [1e-4, -0.999, 0.001],
        [5.0,   0.999, 5.0]
    )

    try:
        params, _ = curve_fit(
            ssvi_total_variance,
            k,
            w,
            p0=[theta0, rho0, phi0],
            bounds=bounds,
            maxfev=20000
        )
    except RuntimeError:
        return None

    def iv_func(K):
        k = np.log(K / F)
        w = ssvi_total_variance(k, *params)
        return np.sqrt(np.maximum(w, 0) / T)

    return iv_func

# ======================
# LOAD DATA
# ======================

ticker = yf.Ticker(TICKER) #Initialising ticker
spot = ticker.history(period="1d")["Close"].iloc[-1] #Retrieving last closing price and using it as the stock price

today = pd.Timestamp.today().normalize() #Retrieving todays date and setting the clock to 00:00

surface = []

for exp in ticker.options:

    t = (pd.to_datetime(exp)-today).days #Calculating the amount of days until expiry
    
    if t <= MIN_DAYS or t > MAX_DAYS:
        continue
    else:
        T =  t / 365.0 #Turning the amount of days until expiry into years

    opt = ticker.option_chain(exp) #Retrieving call and put options with strike equal chosen expiry
    iv_func = fit_ssvi(opt.calls, opt.puts, spot, T) #Fitting SSVI

    if iv_func is None: #Checking if fit_ssvi returned something
        continue

    K_grid = np.linspace(ITM_FACTOR * spot, OTM_FACTOR * spot, 300) #Creating the Strike grid
    iv_grid = iv_func(K_grid)

    call_prices = np.array([
        bs_call_price(spot, K, T, RISK_FREE_RATE, iv)
        for K, iv in zip(K_grid, iv_grid)
    ])

    # ======================
    # Breeden Litzenberger
    # ======================

    dK = K_grid[1] - K_grid[0]
    d2C = np.gradient(np.gradient(call_prices, dK), dK)
    pdf = np.exp(RISK_FREE_RATE * T) * d2C
    pdf[pdf < 0] = 0

    integral = np.trapezoid(pdf, K_grid) 
    if integral <= 0:
        continue

    pdf /= integral #Normalising
    pdf = gaussian_filter1d(pdf, SMOOTH_SIGMA) #Using a gaussian filter for smoothing

    expected_forward = np.trapezoid(K_grid * pdf, K_grid)
    theoretical_forward = spot * np.exp(RISK_FREE_RATE * T)

    if abs(expected_forward - theoretical_forward) / spot > FORWARD_TOL:
        continue

    surface.append((T * 365, K_grid, pdf))

# ======================
# DASH APP
# ======================

app = dash.Dash(__name__)

fig = go.Figure()

for days, K, pdf in surface:
    fig.add_trace(go.Scatter3d(
        x=K,
        y=[days] * len(K),
        z=pdf,
        mode="lines"
    ))

fig.update_layout(
    title="NOC Risk-Neutral Density Surface (SSVI)",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Days to Expiry",
        zaxis_title="Density"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

app.layout = html.Div(
    style={"width": "100vw", "height": "100vh"},
    children=[
        dcc.Graph(
            figure=fig,
            style={"width": "100%", "height": "100vh"}
        )
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
