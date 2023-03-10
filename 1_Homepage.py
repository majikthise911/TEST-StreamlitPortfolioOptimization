import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="π",
)

st.title("Welcome")
st.sidebar.success("Select a page above.")

# if "my_input" not in st.session_state:
#     st.session_state["my_input"] = ""

# my_input = st.text_input("Input a text here", st.session_state["my_input"])
# submit = st.button("Submit")
# if submit:
#     st.session_state["my_input"] = my_input

st.markdown('''π° Are you tired of loosing big by being a short-term gambler in the stock/crypto market π°

Looking like a foolπ€¦ββοΈ at Thanksgiving ππ dinner after you convinced your Uncle and Grandmother to invest their savings into shit coinsπ© and WSB's Stonks π at peak market prices? 

Ooof...**Powerful** πͺ

Welcome to **Wealthwise**, a financial planning and investment management application for long-term investors! π

With Wealthwise, you can analyze your current portfolio and or test out new ones using **Modern Portfolio Theory** or **MPT** for short. 

Our platform is specifically designed to optimize your dollar cost averaging strategies, giving you the best chance at success in the long run. π

All in the name of building a brighter financial future, or whatever. Give it a try, I guess... π€·ββοΈ

''')

# Hide Streamlit Menu and Footer
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)