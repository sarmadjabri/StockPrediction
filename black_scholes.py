import numpy as np
from scipy.stats import norm
from logging_config import setup_logging
setup_logging()
import logging

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate the Black-Scholes option price."""
    logging.info(f"Calculating Black-Scholes for S={S}, K={K}, T={T}, r={r}, sigma={sigma}, option_type={option_type}.")
    
    try:
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            logging.error("Invalid option type. Must be 'call' or 'put'.")
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        
        logging.info(f"Black-Scholes price calculated: {price}")
        return price
    except Exception as e:
        logging.error(f"Error calculating Black-Scholes price: {e}")
        return None
