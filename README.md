# SSVI Risk-Neutral PDF Model

This project models the **risk-neutral probability density function (PDF)** of option prices across strikes and maturities using a smooth, arbitrage-free volatility surface.

The model is applied to option data for **:contentReference[oaicite:0]{index=0} (ticker: NOC)**.

---

## How It Works

The script constructs a **3D surface of the risk-neutral PDF** as a function of strike price and time to maturity.

### Data
- Option data is retrieved from **Yahoo Finance** using the `yfinance` API.

### Methodology
The model combines the following components:

- **Stochastic Simple Variance Interpolation (SSVI)**  
  Used to model a smooth, arbitrage-free implied volatility surface.

- **Black–Scholes (BS)**  
  Used to price options based on the smoothed implied volatility surface.

- **Breeden–Litzenberger (BL)**  
  Used to derive the risk-neutral PDF by taking the second derivative of option prices with respect to strike.

Because the BL formula involves second-order derivatives, it is highly sensitive to noise.  
Therefore, a smooth option pricing model (via BS + SSVI) is essential, at least with volume as low as observed here.

### Assumptions and Limitations
- All models used are theoretically valid for **European options**.
- Although NOC options are American, the dividend yield is sufficiently low that early exercise is considered highly unlikely.
- The model is intended for **illustrative and analytical purposes only**, not for price prediction.

---

## Notes
This project was developed with the assistance of **AI-based tools** for coding support and iteration.  

