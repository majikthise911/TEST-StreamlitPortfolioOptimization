import streamlit as st
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from streamlit_option_menu import option_menu # pip install streamlit-option-menu
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import os
import copy
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import logging



	
# -------------- PAGE CONFIG --------------
page_title = "Financial Portfolio Optimizer"
page_icon = ":moneybag:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"

st.set_page_config(page_title = page_title, layout = layout, page_icon = page_icon)
st.title(page_title + " " + page_icon)

######################

selected = option_menu(
    menu_title=None,
    options=["MPT Analysis", "Theory"], 
    icons=["bar-chart-fill", "pencil-fill"], #https://icons.getbootstrap.com/
    orientation="horizontal",
)

if selected == "Theory":
	st.header("Theory")
	st.write('''
	Modern Portfolio Theory (MPT) is a framework for constructing investment portfolios that aims to maximize expected return 
	for a given level of risk. It is based on the idea that investors are risk-averse, meaning that they prefer a certain level 
	of return to a higher level of risk. MPT helps investors balance their desire for high returns with their need to minimize risk 
	by diversifying their investments across multiple asset classes.

	The core concept of MPT is the efficient frontier, which is a curve that represents the highest expected return for a given level 
	of risk. The efficient frontier is constructed by plotting the returns and standard deviations of all possible portfolios, and 
	selecting those portfolios that lie on the curve. These portfolios are considered efficient because they offer the highest expected 
	return for a given level of risk.

	MPT is based on several key assumptions:	

	1. Investors are rational and seek to maximize their expected utility. 
	2. Investors are risk-averse, meaning that they prefer a certain level of return to a higher level of risk.
	3. Markets are efficient, meaning that asset prices reflect all available information.
	4. There is a positive correlation between risk and return.
	5. Returns are normally distributed.

	One of the key tools used in MPT is the capital asset pricing model (CAPM), which is used to calculate the expected return of 
	an investment. The CAPM takes into account the risk-free rate (the return on a risk-free investment such as a U.S. Treasury bond) and 
	the market risk premium (the difference between the expected return on the market and the risk-free rate).

	MPT is often used by financial advisors and investors to construct portfolios that are tailored to meet specific investment 
	goals. It is important to note, however, that MPT is based on several assumptions that may not hold true in all circumstances. 
	In practice, investors may need to consider other factors such as taxes, transaction costs, and personal preferences when constructing 
	their portfolios.

	In summary, Modern Portfolio Theory is a framework for constructing investment portfolios that aims to maximize expected return for a 
	given level of risk. It is based on the assumption that investors are risk-averse and seek to maximize their expected utility, and that 
	markets are efficient and prices reflect all available information. While MPT can be a useful tool for investors, it is important to consider 
	a range of factors when constructing a portfolio.

	''')

	
if selected == "MPT Analysis":
	st.header("MPT Analysis")

####################

col1, col2 = st.columns(2)

with col1:
	start_date = st.date_input("Start Date",datetime(2013, 1, 1))
	
with col2:
	end_date = st.date_input("End Date") # it defaults to current date

tickers_string = st.text_input('Enter all stock tickers to be included in portfolio separated by commas \
								WITHOUT spaces, e.g. "TSLA,AAPL,MSFT,ETH-USD,BTC-USD,MATIC-USD,GOOG"', 'TSLA,AAPL,MSFT,ETH-USD,BTC-USD,MATIC-USD,GOOG').upper()
tickers = tickers_string.split(',')

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------- PAGE CONFIG --------------

# -------------- FUNCTIONS ----------------
def plot_cum_returns(data, title):    
	daily_cum_returns = 1 + data.dropna().pct_change()
	daily_cum_returns = daily_cum_returns.cumprod()*100
	fig = px.line(daily_cum_returns, title=title)
	return fig
	
def plot_efficient_frontier_and_max_sharpe(mu, S): # mu is expected returns, S is covariance matrix. So we are defining a function that takes in these two parameters
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S) # the efficient frontier object is 
	fig, ax = plt.subplots(figsize=(6,4)) # fig, ax = plt.subplots() is the same as fig = plt.figure() and ax = fig.add_subplot(111)
	ef_max_sharpe = ef.deepcopy() # there are different ways to do this, like copy.deepcopy(ef), but this other way breaks the code on cloud deployment
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0.02)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

# The code to get stock prices using yfinance is below and in a try/except block because it sometimes fails and we need to catch the error
try:
	# Get Stock Prices using pandas_datareader Library	
	### stocks_df = DataReader(tickers, 'yahoo', start = start_date, end = end_date)['Adj Close']	
	stocks_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
	# Plot Individual Stock Prices
	fig_price = px.line(stocks_df, title='Price of Individual Stocks')
	# Plot Individual Cumulative Returns
	fig_cum_returns = plot_cum_returns(stocks_df, 'Cumulative Returns of Individual Stocks Starting with $100')
	# Calculatge and Plot Correlation Matrix between Stocks
	corr_df = stocks_df.corr().round(2) # round to 2 decimal places
	fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
		
	# Calculate expected returns and sample covariance matrix for portfolio optimization later
	mu = expected_returns.mean_historical_return(stocks_df)
	S = risk_models.sample_cov(stocks_df)

	# Plot efficient frontier curve
	fig = plot_efficient_frontier_and_max_sharpe(mu, S)
	fig_efficient_frontier = BytesIO()
	fig.savefig(fig_efficient_frontier, format="png")

	# Get optimized weights
	ef = EfficientFrontier(mu, S)
	ef.max_sharpe(risk_free_rate=0.02)
	weights = ef.clean_weights()
	expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
	weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
	weights_df.columns = ['weights']

	# Calculate returns of portfolio with optimized weights
	stocks_df['Optimized Portfolio'] = 0
	for ticker, weight in weights.items():
		stocks_df['Optimized Portfolio'] += stocks_df[ticker]*weight

	# # Plot Cumulative Returns of Optimized Portfolio
	fig_cum_returns_optimized = plot_cum_returns(stocks_df['Optimized Portfolio'], 'Cumulative Returns of Optimized Portfolio Starting with $100')
# -------------- FUNCTIONS ----------------

# -------------- STREAMLIT APP ------------
	# # Display everything on Streamlit
	st.subheader("Your Portfolio Consists of {} Stocks".format(tickers_string))	
	st.plotly_chart(fig_cum_returns_optimized)

	st.subheader("Optimized Max Sharpe Portfolio Weights")
	st.dataframe(weights_df)

	st.subheader("Optimized MaxSharpe Portfolio Performance")
	st.image(fig_efficient_frontier)

	st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
	st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
	st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))

	st.plotly_chart(fig_corr) # fig_corr is not a plotly chart
	st.plotly_chart(fig_price)
	st.plotly_chart(fig_cum_returns)
except:
	st.write(logging.exception(''))
	#st.write('Enter correct stock tickers to be included in portfolio separated\
# by commas WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"and hit Enter.')	

# Create a function to download the weights_df dataframe as a csv file using a button
@st.cache # this is a decorator that caches the function so that it doesn't have to be rerun every time the app is run
def convert_df(weights_df): # this function converts the weights_df dataframe to a csv file
    # code to create or retrieve the weights_df dataframe goes here
    return weights_df.to_csv().encode('utf-8')
csv = convert_df(weights_df) # assign the output of the convert_df function to a variable called csv

st.download_button( # this creates a download button
	label="Download Optimized Weights as CSV",
	data=csv,
	file_name='weights_df.csv',
	mime='text/csv'
)
# -------------- STREAMLIT APP ------------

#------------------------TEST-------------------------

# embed a loom video 

# TODO: Add a button to refresh the page+
# TODO: add input for risk_free_rate to be changed by user 
# TODO: Add a button to download the optimized portfolio weights - Completed
# TODO: Change how the optimized portfolio list is sorted. Sort by weights instead of alphabetically - Completed
# TODO: Add section to have GPT-3 generate a stock portfolio
# TODO: Add section to have GPT-3 generate a report on the portfolio
# TODO: Add button to each graph that shows the code/math to generate the graph
#		Maybe find a way to integrate chat bot to explain the math/code with the chat bot that 
# 		I already built in streamlit 