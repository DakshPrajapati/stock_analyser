from matplotlib import layout_engine
import streamlit as st

import pandas as pd
import numpy as np
import yfinance as yf
from numpy import array

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from itertools import cycle
from decimal import Decimal

from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from xgboost import XGBRegressor
from sklearn.svm import SVR


original_title = '<p style="font-family:Courier; color:rgb(140, 255, 107); font-size: 56px; font-weight : 700"><*> Stock Analyzer </p>'
st.markdown(original_title, unsafe_allow_html=True)

st.text("")
st.text("")
st.header("\n:wave: Hola,")

st.subheader("We welcome you here,  Now let's make some money !")
st.text("")
st.text("")
st.text("")

st.write("- :smile:", " ","We analyse any Stock that you want to on the basis of current market movement and News that are coming out. ")

st.write("- We do this by using machine learning algorithms that help us predict future movement :robot_face:.")

st.write("- Now you are not needed to visit various different websites to get your information... ALL OF THAT CAN BE DONE IN ONE CLICK.")

st.text("")
st.text("")
st.text("")

st.subheader("Let's Begin :handshake:")
st.text("")
name = st.text_input('Enter the ticker sysmbol of the Stock  (eg. MSFT for Microsoft)',)

# Indian = st.checkbox('is it a Indian stock?')

def dataCleaning(ticker):
    
    ticker.reset_index(inplace=True)

    ticker.rename(columns={"Date":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close"}, inplace=True)
    ticker.dropna()

    ticker.date = pd.to_datetime(ticker.date)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

technical = 1


if name:

    st.text("")
    st.text("")

    st.write("* Dataset on which Machine Learning model is begin trained ")
    st.text("")
    
    ticker = yf.download(name, start="2022-01-01")
    
    dataCleaning(ticker)

    st.dataframe(ticker.style.highlight_max(axis=0))

    closedf = ticker[['date','close']]

    close_stock = closedf.copy()
    del closedf['date']
    
    scaler=MinMaxScaler(feature_range=(0,1))
    
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

    training_size=int(len(closedf)*0.65)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    svr_rbf = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
    svr_rbf.fit(X_train, y_train)

    train_predict=svr_rbf.predict(X_train)
    test_predict=svr_rbf.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)
    
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 10
    while(i<pred_days):
    
        if(len(temp_input)>time_step):
        
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
        
            yhat = svr_rbf.predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat.tolist())
            temp_input=temp_input[1:]
       
            lst_output.extend(yhat.tolist())
            i=i+1
        
        else:
            yhat = svr_rbf.predict(x_input)
        
            temp_input.extend(yhat.tolist())
            lst_output.extend(yhat.tolist())
        
            i=i+1
        
    print("Output of predicted next days: ", len(lst_output))


    last_days=np.arange(1,time_step+1)
    day_pred=np.arange(time_step+1,time_step+pred_days+1)
    print(last_days)
    print(day_pred)


    temp_mat = np.empty((len(last_days)+pred_days+1,1))
    temp_mat[:] = np.nan
    temp_mat = temp_mat.reshape(1,-1).tolist()[0]

    last_original_days_value = temp_mat
    next_predicted_days_value = temp_mat

    last_original_days_value[0:time_step+1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
    next_predicted_days_value[time_step+1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print((last_original_days_value[20]))
    print((last_original_days_value[14]))

    diff = last_original_days_value[20] - last_original_days_value[14]

    st.text("")

    min = max = last_original_days_value[16]

    for i in range(16,25):
        if last_original_days_value[i]<min : min = last_original_days_value[i]
        if last_original_days_value[i]>max : max = last_original_days_value[i]

    if diff>0:
        st.subheader("	:large_green_circle: We expect to see a rise in price in following days.. :chart_with_upwards_trend:")
        st.text("We suggest you to buy the stock")
        st.write("You can traget upto - ",max)
    else:

        st.subheader(" :red_circle: Price may fall in comming days.. :chart_with_downwards_trend:")
        st.write("- We suggest you to short the stock")
        st.write("- You can traget upto - ",min)
    st.write("")
    st.write("")
    st.subheader('Price for tommorow can be ')
    st.subheader(last_original_days_value[16])
    # print(next_predicted_days_value)

    new_pred_plot = pd.DataFrame({
        'last_original_days_value':last_original_days_value,
        'next_predicted_days_value':next_predicted_days_value
    })
    

    names = cycle([''])

    fig = px.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                          new_pred_plot['next_predicted_days_value']],
                  labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(
                      paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', font_size=15, font_color='black',legend_title_text='')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.write("")
    st.write("")

    st.write('- Compare last 15 days vs next 10 days')

    st.plotly_chart(fig, use_container_width=True)
    #fig.show()


    svrdf=closedf.tolist()
    svrdf.extend((np.array(lst_output).reshape(-1,1)).tolist())
    svrdf=scaler.inverse_transform(svrdf).reshape(1,-1).tolist()[0]

    names = cycle([''])

    fig = px.line(svrdf,labels={'value': 'Stock price','index': 'Timestamp'})
    fig.update_layout(
                      paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', font_size=15, font_color='black',legend_title_text='')
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.write("")
    st.write("")

    st.write('- Plotting whole closing stock price with prediction')

    st.plotly_chart(fig, use_container_width=False)

##################################################################


    st.text("")
    st.text("")
    st.subheader(" :newspaper: Now lets check the sentiment")
    st.text("")
    st.text("")

    finwiz_url = "https://finviz.com/quote.ashx?t="    
    url = finwiz_url + (name)


    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)
    
    
    html = BeautifulSoup(response, features="lxml")


    news_table = html.find(id='news-table')
    news_table_tr = news_table.find_all('tr')

    parsed_news = []

    for x in news_table.findAll('tr'):
        
        text = x.a.get_text() 
        date_scrape = x.td.text.split()
        
        if len(date_scrape) == 1:
            time = date_scrape[0]
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        parsed_news.append([ date, time, text])

    
    vader = SentimentIntensityAnalyzer()


    columns = ['date', 'time', 'headline']
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)
    
    
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    
    
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

    st.write("- the following table shows news from different sources that is coming out")
    st.write("- We are getting a general understanding from those headlines by performing NLP :mag:")

    st.text("")
    st.text("")

    st.dataframe(parsed_and_scored_news.style.highlight_max(axis=0))


    sumOfRecentCompound = parsed_and_scored_news['compound'].head().sum()

    st.text("")
    st.text("")


    st.write(":scroll: After Analysing recent news, score comes out to be")
    st.subheader(sumOfRecentCompound)

    sentiment = 1

    if sumOfRecentCompound > -1 and sumOfRecentCompound < 1:
        st.write("Market does not lean towards any side :neutral_face:")
    elif sumOfRecentCompound > 1:
        st.write("Market is falling positive about this stock :heart_eyes:")
        sentiment = 2
    elif sumOfRecentCompound < -1:
        st.write("Overall sentiment about this stock is negetive :fearful:")
        sentiment = 0

    st.text("")
    st.text("")

    st.header("	:moneybag: Final verdict")

    if diff > 0:
        if sentiment == 2:
            st.write('Both fundamentals and sentiments are suggesting an upwards trend')
        elif sentiment == 1:
            st.write('Eventhough sentiments are neutral, we suggest keep this stock in your rader as fundamentals good and a trendchange maybe on its way.')
        elif sentiment == 0:
            st.write("You should wait to see a trend change as fundamentals and sentiments are not lineing up")
    else:
        if sentiment == 2:
            st.write('You should wait to see a trend change as fundamentals and sentiments are not lineing up')
        elif sentiment == 1:
            st.write('Eventhough sentiments are neutral, we suggest you to keep your distance from this stock or short it if you can.')
        elif sentiment == 0:
            st.write("Definately a great oppertunity to short the stock")

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    st.subheader('	:thumbsup: Thanks for visiting... We hope this was helpfull')
    st.write('- made by Daksh Prajapati(19dcs110) and Daksh Patel(19dcs083) :smile::smile:')