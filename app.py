import streamlit as st 
from streamlit_option_menu import option_menu
st.logo("logo1.jpg",icon_image="logo1.jpg")
selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "📈 Prediction", "📝 Feedback","🔐 Admin"],
    icons=["🏠","📈","📝","🔐"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)
if selected == "🏠 Home":
    import streamlit as st
    import plotly.express as px
    import os
    import numpy as np      # To use np.arrays
    import pandas as pd     # To use dataframes
    from pandas.plotting import autocorrelation_plot as auto_corr

    # To plot
    import matplotlib as mpl
    import seaborn as sns
    import matplotlib.pyplot as plt  
    mpl.pyplot.ion()

    #For date-time
    import math
    import datetime
    import calendar
    from datetime import datetime
    from datetime import timedelta

    # Another imports if needs
    import itertools
    import statsmodels.api as sm
    import statsmodels.tsa.api as smt
    import statsmodels.formula.api as smf

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression 
    from sklearn import preprocessing

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima_model import ARIMA
    from pmdarima.utils import decomposed_plot
    from pmdarima.arima import decompose
    from pmdarima import auto_arima

    import warnings
    warnings.filterwarnings("ignore")
    pd.options.display.max_columns=100 # to see columns 

    df_store = pd.read_csv('stores.csv') #store data
    df_train = pd.read_csv('train.csv') # train set
    df_features = pd.read_csv('features.csv') #external information

    df_store['Type'] = df_store['Type'].replace({'A': 'Walmart Supercenter', 'B': 'Walmart Discount Store', 'C': 'Walmart Neighborhood Market'})

    # merging 3 different sets
    df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
    #print(df.shape)

    df.drop(['IsHoliday_y'], axis=1,inplace=True)
    df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True)
    df = df.loc[df['Weekly_Sales'] > 0]

    first_and_last_5 = df['Date'].head(5)._append(df['Date'].tail(5))

    df.sort_values(by='Weekly_Sales',ascending=False).head(5) #5 highest weekly sales

    df = df.fillna(0) # filling null's with 0
    #print(df.isna().sum()) #Last null check

    df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
    #df['week'] =df['Date'].dt.week
    df['week'] = df['Date'].dt.isocalendar().week
    #df['month'] =df['Date'].dt.month
    df['month'] = df['Date'].dt.strftime('%B')
    #df['month'] = df['Date'].dt.to_period('M').dt.month_name()
    #df['month'] = df['Date'].dt.to_period('M').map(lambda x: x.month())
    df['year'] =df['Date'].dt.year

    df.to_csv('clean_data.csv') # assign new data frame to csv for using in Streamlit webAPP

    def mae_test(test, pred):
        error = np.mean(np.abs(test - pred))
        return error

    pd.options.display.max_columns=100 # to see columns 
    df = pd.read_csv('clean_data.csv')
    df.drop(columns=['Unnamed: 0'],inplace=True)
    df['Date'] = pd.to_datetime(df['Date']) # changing datetime to divide if needs

    ### Time Series Model ###

    df["Date"] = pd.to_datetime(df["Date"]) #changing data to datetime for decomposing
    df.set_index('Date', inplace=True) #seting date as index

    df_week = df.select_dtypes(include=[np.number]).resample('W').mean()
    #df_week = df.resample('W').mean() #resample data as weekly
    #df_week = df['Weekly_Sales'].resample('W').mean()

    ### Train - Test Split of Weekly Data ###

    train_data = df_week[:int(0.7*(len(df_week)))] 
    test_data = df_week[int(0.7*(len(df_week))):]
    #st.title("Walmart Prediction EDA")
    #st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)
    logo_url = "https://paymentsnext.com/wp-content/uploads/2017/10/Walmart-logo.png"  # Replace with your logo URL or file path
    st.markdown(f'''
        <div class="header">
            <img src="{logo_url}" width="60" height="60">
            <h1 class="hover-effect"> Walmart Prediction EDA</h1>
        </div>
        ''', unsafe_allow_html=True)
    # CSS for hover effect
    st.markdown(
        """
        <style>
        .hover-effect:hover {
            color: blue;
            font-size: 2.5em;
            transition: 0.3s;
        }
        .header {
            display: flex;
            align-items: center;
        }
        .header img {
            margin-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    #os.chdir(r"C:\Python312\Walmart Sales")
    df = pd.read_csv("clean_data.csv", encoding = "ISO-8859-1")

    col1, col2 = st.columns((2))
    df["Date"] = pd.to_datetime(df["Date"])

    # Getting the min and max date 
    startDate = pd.to_datetime(df["Date"]).min()
    endDate = pd.to_datetime(df["Date"]).max()

    with col1:
        date1 = pd.to_datetime(st.date_input("Start Date", startDate))

    with col2:
        date2 = pd.to_datetime(st.date_input("End Date", endDate))

    df = df[(df["Date"] >= date1) & (df["Date"] <= date2)].copy()

    st.sidebar.header("Choose your filter: ")
    #Create for Types
    type_s = st.sidebar.multiselect("Choose Type of Sales", df["Type"].unique())
    if not type_s:
        df2 = df.copy()
    else:
        df2 = df[df["Type"].isin(type_s)]


    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    mon_s = st.sidebar.multiselect("Choose Month of Sales", months)
    if not mon_s:
        df3 = df2.copy()
    else:
        month_numbers = [months.index(month)+1 for month in mon_s]
        df3 = df2[df2["month"].isin(month_numbers)]

    year_s = st.sidebar.multiselect("Choose Year of Sales",df3["year"].unique())

    # Filter the data based on Type, Month and Year

    if not type_s and not mon_s and not year_s:
        filtered_df = df
    elif not mon_s and not year_s:
        filtered_df = df[df["Type"].isin(type_s)]
    elif not type_s and not year_s:
        filtered_df = df[df["month"].isin(mon_s)]
    elif mon_s and year_s:
        filtered_df = df3[df["month"].isin(mon_s) & df3["year"].isin(year_s)]
    elif type_s and year_s:
        filtered_df = df3[df["Type"].isin(type_s) & df3["year"].isin(year_s)]
    elif type_s and mon_s:
        filtered_df = df3[df["Type"].isin(type_s) & df3["month"].isin(mon_s)]
    elif year_s:
        filtered_df = df3[df3["year"].isin(year_s)]
    else:
        filtered_df = df3[df3["Type"].isin(type_s) & df3["month"].isin(mon_s) & df3["year"].isin(year_s)]

    dept_df = filtered_df.groupby(by = ["Dept"], as_index = False)["Weekly_Sales"].sum()

    with col1:
        st.subheader("Department wise Sales")
        fig = px.bar(dept_df, x = "Dept", y = "Weekly_Sales", text = ['${:,.2f}'.format(x) for x in dept_df["Weekly_Sales"]],
                    template = "seaborn")
        st.plotly_chart(fig,use_container_width=True, height = 200)

    with col2:
        st.subheader("Month wise Sales")
        fig = px.pie(filtered_df, values = "Weekly_Sales", names = "month", hole = 0.5)
        fig.update_traces(textinfo = "label", text = filtered_df["month"], textposition = "outside")
        st.plotly_chart(fig,use_container_width=True)

    st.subheader('Store wise Sales')
    store_df = filtered_df.groupby(by = ["Store"], as_index = False)["Weekly_Sales"].sum()
    #linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Weekly_Sales"].sum()).reset_index()
    figx = px.line(store_df, x = "Store", y="Weekly_Sales", labels = {"Weekly_Sales": "Sales"},height=500, width = 1000,template="gridon")
    st.plotly_chart(figx,use_container_width=True)

    filtered_df["month_year"] = filtered_df["Date"].dt.to_period("M")
    st.subheader('Time Series Analysis')

    linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Weekly_Sales"].sum()).reset_index()
    fig2 = px.line(linechart, x = "month_year", y="Weekly_Sales", labels = {"Weekly_Sales": "Amount"},height=500, width = 1000,template="gridon")
    st.plotly_chart(fig2,use_container_width=True)

    # st.subheader('Prediction of Weekly Sales Using Auto-ARIMA')

    ### Train Test Split For Auto-Arima Model ###
    df_week_diff = df_week['Weekly_Sales'].diff().dropna() #creating difference values
    train_data_diff = df_week_diff [:int(0.7*(len(df_week_diff )))]
    test_data_diff = df_week_diff [int(0.7*(len(df_week_diff ))):]

    st.subheader('Prediction of Weekly Sales Using Exponential Smoothing')

    model_holt_winters = ExponentialSmoothing(train_data_diff, seasonal_periods=20, seasonal='additive',
                                            trend='additive',damped=True).fit() #Taking additive trend and seasonality.
    y_pred = model_holt_winters.forecast(len(test_data_diff))# Predict the test data

    st.sidebar.header("Predicted Data")
    #date_input = st.sidebar.date_input("Enter a date:", value=pd.to_datetime("2012-10-26"))
    date_input = st.sidebar.date_input("Enter a date:", 
                                    value=pd.to_datetime("2012-10-26"), 
                                    min_value=pd.to_datetime("2010-02-01"), 
                                    max_value=pd.to_datetime("2025-12-31"))

    # Convert the date input to an integer index
    #date_index = (date_input - train_data_diff.index[0]).days // 7
    #date_index = (pd.to_datetime(date_input) - train_data_diff.index[0]).days // 7
    date_index = min((pd.to_datetime(date_input) - train_data_diff.index[0]).days // 7, len(y_pred) - 1)

    # Get the predicted value for the selected date
    pred_value = y_pred[date_index]

    fig = px.line(x=train_data_diff.index, y=train_data_diff.values, labels={'x': 'Date', 'y': 'Weekly Sales'}, height=500, width=1000, template="gridon")
    fig.add_scatter(x=train_data_diff.index, y=train_data_diff.values, mode='lines', name='Train', line=dict(color='blue'))
    fig.add_scatter(x=test_data_diff.index, y=test_data_diff.values, mode='lines', name='Test')
    fig.add_scatter(x=y_pred.index, y=y_pred.values, mode='lines', name='Prediction of ExponentialSmoothing')
    fig.add_scatter(x=[date_input], y=[pred_value], mode='markers', name='Predicted Value', marker=dict(color='red', size=10))
    st.plotly_chart(fig, use_container_width=True)
    
if selected == "📈 Prediction":
    st.header("Walmart Sales Prediction using LSTM")
    import streamlit as st
    import pandas as pd
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    import os
    import plotly.graph_objects as go
    import plotly.express as px
    import base64

    # Constants
    N_STEPS = 3
    N_FEATURES = 1
    EPOCHS = 200

    def create_pie_chart(sales_data):
        fig = px.pie(sales_data, values='Net_sales', names=sales_data.index.year, hole=0.5)
        fig.update_traces(textinfo='percent+label', hoverinfo='label+percent+value', textposition="outside")
        return fig

    def load_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def preprocess_data(sales_data: pd.DataFrame) -> pd.DataFrame:
        sales_data.index = pd.to_datetime(sales_data['Year'], format='%Y')
        sales_data = sales_data.drop('Year', axis=1)
        return sales_data

    def prepare_data(timeseries_data: pd.Series, n_steps: int) -> tuple:
        X, y = [], []
        for i in range(len(timeseries_data)):
            end_ix = i + n_steps
            if end_ix > len(timeseries_data) - 1:
                break
            seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def create_and_train_model(X: np.ndarray, y: np.ndarray) -> Sequential:
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(N_STEPS, N_FEATURES)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=EPOCHS, verbose=1)
        return model

    def predict_next_years(model: Sequential, n_steps: int, n_features: int, n_predictions: int, timeseries_data: pd.Series) -> list:
        predictions = list(timeseries_data[-n_steps:])
        for _ in range(n_predictions):
            x_input = np.array(predictions[-n_steps:]).reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            predictions.append(yhat[0][0])
        return predictions[-n_predictions:]
    logo_url = "https://paymentsnext.com/wp-content/uploads/2017/10/Walmart-logo.png"  # Replace with your logo URL or file path
    st.markdown(f'''
        <div class="header">
            <img src="{logo_url}" width="60" height="60">
            <h1 class="hover-effect">Walmart Sales Forecast</h1>
        </div>
        ''', unsafe_allow_html=True)

    # CSS for hover effect
    st.markdown(
        """
        <style>
        .hover-effect:hover {
            color: blue;
            font-size: 2.5em;
            transition: 0.3s;
        }
        .header {
            display: flex;
            align-items: center;
        }
        .header img {
            margin-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        Walmart Inc. is an American multinational retail corporation that operates a chain of hypermarkets, discount department stores, and grocery stores. 
        Founded by Sam Walton in 1962, Walmart has grown to become one of the largest companies in the world by revenue. 
        The company is known for its wide range of products at low prices, making it a popular shopping destination for millions of customers.
        """
    )

    sales_data = load_data("Walmart_NetSales.csv")
    sales_data = preprocess_data(sales_data)
    timeseries_data = sales_data['Net_sales']

    X, y = prepare_data(timeseries_data, N_STEPS)
    X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))

    model = create_and_train_model(X, y)

    st.subheader("Historical Sales Proportion by Year")
    st.plotly_chart(create_pie_chart(sales_data), use_container_width=True)

    st.markdown(f'''
            <div class="header">
                <h1 class="hover-effect">Prediction of Yearly Sales Using LSTM</h1>
            </div>
            ''', unsafe_allow_html=True)

    num_years = st.selectbox("Upto which year do you want to predict next?", range(2026, 2035), index=4) - 2024

    next_years = predict_next_years(model, N_STEPS, N_FEATURES, num_years, timeseries_data)

    st.subheader(f"Predictions for the next {num_years} years:")
    for i in range(num_years):
        st.write(f"Prediction for year {2025 + i}: {next_years[i]:.2f} in billion U.S. dollars")

    st.success("Prediction completed successfully!")

    years = list(sales_data.index.year) + [year for year in range(sales_data.index.year[-1] + 1, sales_data.index.year[-1] + num_years + 1)]

    fig = go.Figure()

    all_years = years
    all_sales = list(sales_data['Net_sales']) + list(next_years)

    fig.add_trace(go.Bar(x=years[:len(sales_data)], y=sales_data['Net_sales'],
                        name='Historical Sales',
                        hovertemplate='Year: %{x}<br>Sales: $%{y:.2f} billion<extra></extra>',
                        marker_color='blue', opacity=0.6))

    fig.add_trace(go.Bar(x=years[len(sales_data):], y=next_years,
                        name='Predicted Sales',
                        hovertemplate='Year: %{x}<br>Predicted Sales: $%{y:.2f} billion<extra></extra>',
                        marker_color='orange', opacity=0.6))

    fig.add_trace(go.Scatter(x=years[:len(sales_data)], y=sales_data['Net_sales'],
                            mode='lines+markers', name='Historical Trend',
                            line=dict(color='rgba(221,170,221,0.7)'),
                            hovertemplate='Year: %{x}<br>Sales: $%{y:.2f} billion<extra></extra>'))

    fig.add_trace(go.Scatter(x=years[len(sales_data):], y=next_years,
                            mode='lines+markers', name='Predictions',
                            line=dict(dash='dash', color='red'),
                            hovertemplate='Year: %{x}<br>Predicted Sales: $%{y:.2f} billion<extra></extra>'))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Net Sales (in billion U.S. dollars)',
        hovermode='x unified',
        barmode='overlay'
    )

    fig.add_vline(x=years[len(sales_data)-1] + 0.5, line_dash="dash", line_color="green")

    st.subheader("Sales Forecast Plot")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="Download Predicted Sales Data as CSV",
        data=pd.DataFrame({'Year': years, 'Net Sales': all_sales}).to_csv(index=False),
        file_name='predicted_sales.csv',
        mime='text/csv'
    )


#feedback form

if selected == "📝 Feedback":
    import streamlit as st
    import sqlite3
    import re
    import base64
    @st.cache_data
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Connect to the SQLite database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Create the contacts table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS contacts (id INTEGER PRIMARY KEY, name TEXT, email TEXT, message TEXT)''')

    # Add a contact page
    st.title("Feedback Form")

    # Define CSS styling
    contact_form_style = """
    <style>
    .contact-form {
        max-width: 500px;
        margin: 0 auto;
        padding: 20px;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .contact-form h3 {
        text-align: center;
        margin-bottom: 20px;
    }
    .contact-form input[type="text"], .contact-form input[type="email"], .contact-form textarea {
        width: 100%;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.1);
        resize: none;
    }
    .contact-form input[type="submit"] {
        width: 100%;
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .contact-form input[type="submit"]:hover {
        background-color: #3e8e41;
    }
    </style>
    """

    # Add CSS styling to the page
    st.markdown(contact_form_style, unsafe_allow_html=True)

    # Add a container for the contact form
    contact_form = st.container()

    with contact_form:
        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")

            if st.form_submit_button("Submit"):
                with st.spinner("Sending message..."):
                    # Check if the email is valid
                    if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                        # Insert the user input into the SQLite table
                        c.execute("INSERT INTO contacts (name, email, message) VALUES (?,?,?)", (name, email, message))
                        conn.commit()
                        st.success("Message sent!")
                    else:
                        st.error("Invalid email address")

    # Close the database connection
        conn.close()




#Admin Login Check Feesback 

if selected == "🔐 Admin":
    import streamlit as st
    import sqlite3
    import pandas as pd
    import base64
    import plotly.express as px

    # Set up the database connection
    st.subheader("Welcome to Admin Login Page")
    st.write("Please enter your credentials to login as a Admin.")
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Create the contact messages table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contact_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL
        )
    ''')

    # Set up the login form
    def login_form():
        user_id = 'admin@ml.com'
        password = 'adminlogin'
        entered_user_id = st.text_input('User ID')
        entered_password = st.text_input('Password', type='password')
        if st.button('Login'):
            if entered_user_id == user_id and entered_password == password:
                st.success('Logged in successfully')
                return True
            else:
                st.error('Invalid user ID or password')
                return False
        return False

    # Display the login form
    if not login_form():
        st.stop()

    # Add some CSS to customize the appearance
    st.markdown("""
        <style>
    .main {
                background-color: #f0f0f0;}
   
        .stTable {
                font-size: 12px;
                border-collapse: collapse;
                width: 100%;
            }
        .stTable th,.stTable td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                font-family: Arial, sans-serif; /* Add this line to change the font */
        }
        .stTable th {
                background-color: #f0f0f0;
            }
        .stTable tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        .stTable tr:hover {
                background-color: #ddd;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display the contact messages
    st.title('Feedback Messages')
    # Fetch the contact messages from the database
    cursor.execute('SELECT * FROM contacts')
    messages = cursor.fetchall()
    # Convert the messages to a Pandas DataFrame
    df = pd.DataFrame(messages, columns=['S.No.', 'Name', 'Email', 'Message'])
    # Set the 'S.No.' column as the index
    df.set_index('S.No.', inplace=True)
    # Display the messages in a table
    st.table(df)
    # Close the database connection
    conn.close()
