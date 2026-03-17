import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pandas as pd

def YorkRegression(Xi, Yi, errXi, errYi, iter=100, plot=True):
    """
    Function to perform York linear regression with uncertainties in both X and Y.

    Parameters:
    Xi, Yi : array-like
        Observed values of the independent and dependent variables.
    errXi, errYi : array-like
        Uncertainties associated with Xi and Yi.
    iter : int
        Number of iterations to refine the slope estimate.
    plot : bool
        If True, generates plots comparing OLS and York regression.

    Returns:
    dict
        Regression results including slope, intercept, errors, and statistics.
    """

    Xi = np.array(Xi, dtype=float)
    Yi = np.array(Yi, dtype=float)
    errXi = np.array(errXi, dtype=float)
    errYi = np.array(errYi, dtype=float)

    b_vect = []

    # Initial OLS estimate
    b_ols, a_ols = np.polyfit(Xi, Yi, 1)
    b_vect.append(b_ols)

    wXi = 1 / errXi**2
    wYi = 1 / errYi**2

    # Estimate correlation between errXi and errYi
    with np.errstate(invalid='ignore'):
        ri = np.corrcoef(errXi, errYi)[0, 1]
    if np.isnan(ri):
        ri = 1e-20  # Very small default if correlation is undefined

    alphai = np.sqrt(wXi * wYi)

    # Iterative refinement of the slope
    for i in range(iter):
        b = b_vect[i]
        Wi = (wXi * wYi) / (wXi + b**2 * wYi - 2 * b * ri * alphai)
        X_bar = np.sum(Wi * Xi) / np.sum(Wi)
        Y_bar = np.sum(Wi * Yi) / np.sum(Wi)
        Ui = Xi - X_bar
        Vi = Yi - Y_bar
        BETAi = Wi * ((Ui / wYi) + (b * Vi / wXi) - (b * Ui + Vi) * (ri / alphai))
        b_new = np.sum(Wi * BETAi * Vi) / np.sum(Wi * BETAi * Ui)
        b_vect.append(b_new)

    # Final slope and intercept
    b = b_vect[-1]
    a = Y_bar - b * X_bar
    xi = X_bar + BETAi
    yi = Y_bar + b * BETAi
    x = np.sum(Wi * xi) / np.sum(Wi)
    y = np.sum(Wi * yi) / np.sum(Wi)
    ui = xi - x
    vi = yi - y
    Sb2 = 1 / np.sum(Wi * ui**2)
    Sa2 = (1 / np.sum(Wi)) + (x**2) * Sb2
    S = np.sum(Wi * (Yi - b * Xi - a)**2)
    df = len(Xi) - 2
    pX2 = 1 - chi2.cdf(S, df)

    # Plotting
    if plot:
        plt.figure(figsize=(12, 5))

        # Scatter plot with error bars and regression lines
        plt.subplot(1, 2, 1)
        plt.errorbar(Xi, Yi, xerr=errXi, yerr=errYi, fmt='o', color='black', ecolor='gray', capsize=3)
        x_vals = np.linspace(min(Xi), max(Xi), 100)
        plt.plot(x_vals, a_ols + b_ols * x_vals, label='OLS Regression', color='red')
        plt.plot(x_vals, a + b * x_vals, label='York Regression', color='blue')
        plt.xlabel('Xi')
        plt.ylabel('Yi')
        plt.title('OLS vs York Linear Regression')
        plt.legend()

        # Chi-squared distribution plot
        plt.subplot(1, 2, 2)
        x_chi = np.linspace(chi2.ppf(0.001, df), chi2.ppf(0.999, df), 1000)
        plt.plot(x_chi, chi2.pdf(x_chi, df), color='blue', label='Chi-squared PDF')
        plt.axvline(x=S, color='red', linestyle='--', label='S value')
        plt.title('Chi-squared Distribution')
        plt.xlabel('Chi-squared value')
        plt.ylabel('Probability density')
        plt.legend()

        plt.tight_layout()
        plt.show()

    results = {
        "York Slope": b,
        "York Intercept": a,
        "Expected Xi": xi,
        "Expected Yi": yi,
        "X-residuals": Xi - xi,
        "Y-residuals": Yi - yi,
        "Slope error": np.sqrt(Sb2),
        "Intercept error": np.sqrt(Sa2),
        "S": S,
        "X2-probability": pX2,
        "df": df,
        "Iterated slope values": b_vect
    }

    return results


def RegEqualSigma(Xi, Yi, sXi, sYi, plot=False):
    """
    Performs linear regression on x and y data which has equal measurement uncertainties.
    
    Parameters:
    Xi, Yi : array-like
        Observed values of the independent and dependent variables.
    sXi, sYi : array-like
        Uncertainties associated with Xi and Yi.
    plot : bool
        If True, generates plots of the regression line and the Chi-Squared density function.
        
    Returns:
    dict
        Regression results including slope, intercept, errors, and statistics.
    """
    Xi = np.array(Xi, dtype=float)
    Yi = np.array(Yi, dtype=float)
    sXi = np.array(sXi, dtype=float)
    sYi = np.array(sYi, dtype=float)
    
    N = len(Xi)
    
    # Check if uncertainties are equal across all points
    sigmaEqualityCheck = np.sum((sXi[0] != sXi[1:]) | (sYi[0] != sYi[1:]))
    
    if sigmaEqualityCheck == 0:
        warnsigmaEqualityCheck = " ARE EQUAL. Equal sigma regression is appropriate."
        
        sXi_val = sXi[0]
        sYi_val = sYi[0]
        
        # Computation of the terms of the second degree equation b^2+B*b+C=0
        sum_Xi = np.sum(Xi)
        sum_Yi = np.sum(Yi)
        
        B_num = (-N * np.sum(Yi**2) + sum_Yi**2) * sXi_val**2 - (-N * np.sum(Xi**2) + sum_Xi**2) * sYi_val**2
        B_den = (N * np.sum(Xi * Yi) - sum_Xi * sum_Yi) * sXi_val**2
        B = B_num / B_den
        
        C = -(sYi_val / sXi_val)**2
        
        # Compute Delta of the second degree equation and the slope of the linear regression (b) the positive solution
        delta = B**2 - 4 * C
        b = (-B + np.sqrt(delta)) / 2 
        
        # Compute the intercept (a), the weight (W), the minimized sum of squared weighted regression (S), the Q probability 
        a = (sum_Yi - b * sum_Xi) / N
        W = 1 / (sYi_val**2 + b**2 * sXi_val**2)
        S = W * np.sum((Yi - (a + b * Xi))**2)
        df_val = N - 2
        Q = 1 - chi2.cdf(S, df_val)
        
        # Use the observed points (Xi ,Yi) and W to calculate X_bar and Y_bar
        X_bar = np.sum(W * Xi) / np.sum(W)
        Y_bar = np.sum(W * Yi) / np.sum(W)
        Ui = Xi - X_bar
        Vi = Yi - Y_bar
        wXi = 1 / sXi_val**2
        wYi = 1 / sYi_val**2
        
        alphai = np.sqrt(wXi * wYi)
        ri = 10**-10 # correlation of uncertainties almost equal to zero
        
        BETAi = W * ((Ui / wYi) + ((b * Vi) / wXi) - (b * Ui + Vi) * (ri / alphai))
        xi = X_bar + BETAi # expectation for the Xi values 
        yi = Y_bar + b * BETAi # expectation for the Yi values 
        
        x_val = np.sum(W * xi) / np.sum(W)
        y_val = np.sum(W * yi) / np.sum(W)
        ui = xi - x_val
        vi = yi - y_val
        
        Sb2 = 1 / np.sum(W * ui**2) # error on estimated slope 
        Sa2 = (1 / np.sum(W)) + (x_val**2) * Sb2 # error on estimated intercept

        Sb = np.sqrt(Sb2)
        Sa = np.sqrt(Sa2)
        
        ResXi = Xi - xi # residuals
        ResYi = Yi - yi
        
        MAE_X = np.mean(np.abs(Xi - xi)) # mean absolute error on X
        MAE_Y = np.mean(np.abs(Yi - yi)) # mean absolute error on Y
        MAEdiag = np.sqrt(MAE_X**2 + MAE_Y**2) # diagonal mean absolute error
        
        SSres = np.sum(ResYi**2) # The sum of squares of residuals
        SStot = np.sum((Yi - np.mean(Yi))**2) # The total sum of squares
        
        Rsquared = 1 - (SSres / SStot) # Coefficient of determination
        
    else:
        warnsigmaEqualityCheck = " DIFFERS. You MUST use YorkRegression()"
        b = a = xi = yi = ResXi = ResYi = Rsquared = MAE_X = MAE_Y = MAEdiag = Sb = Sa = S = Q = df_val = None
        df_val = N - 2

    res = {
        "Xi and Yi provided uncertanties:": warnsigmaEqualityCheck,
        "Slope": b,
        "Intercept": a,
        "Expected xi": xi,
        "Expected yi": yi,
        "X-residuals": ResXi,
        "Y-residuals": ResYi,
        "Rsquared": Rsquared,
        "Mean Absolute Error on X": MAE_X,
        "Mean Absolute Error on Y": MAE_Y,
        "Average Mean Absolute Error": MAEdiag,
        "Slope uncertainty": Sb,
        "Intercept uncertainty": Sa,
        "S": S,
        "Q": Q,
        "df": df_val
    }

    if plot and b is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter Plot and Regression Line
        ax1.errorbar(Xi, Yi, xerr=sXi, yerr=sYi, fmt='o', color='black', ecolor='gray', capsize=0)
        x_vals = np.linspace(np.min(Xi), np.max(Xi), 100)
        ax1.plot(x_vals, a + b * x_vals, color='red', label='LSE')
        ax1.set_xlabel('Xi')
        ax1.set_ylabel('Yi')
        ax1.legend(title="Regression Lines")
        
        # Density Plot
        x_chi = np.linspace(max(0.1, chi2.ppf(0.001, df_val)), chi2.ppf(0.999, df_val), 1000)
        ax2.fill_between(x_chi, chi2.pdf(x_chi, df_val), color='blue', alpha=0.5)
        ax2.axvline(x=S, color='red')
        ax2.set_xlabel('X')
        ax2.set_ylabel('density')
        
        plt.tight_layout()
        plt.show()

    return res