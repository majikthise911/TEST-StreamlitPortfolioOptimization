import streamlit as st

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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