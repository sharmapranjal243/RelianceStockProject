import streamlit as st
from PIL import Image
import numpy as np 
import pandas as pd 
import yfinance as yf
import tensorflow as tf
import random as rn
import os
import math
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
warnings.filterwarnings("ignore")


st.title("Reliance Industry Stock Price Forecasting")

image=Image.open("Forecast.jpg")
new_image=image.resize((785,250))
st.image(new_image)

def display_input_data():

    input_details=dict()

    if "start" not in st.session_state:

        st.session_state.start=False

    if "end" not in st.session_state:

        st.session_state.end=False

    def enter_start_date():

        st.session_state.start=True

    def enter_end_date():

        st.session_state.end=True

    start_date=st.sidebar.date_input("Enter Start Date :",on_change=enter_start_date)

    end_date=st.sidebar.date_input("Enter End Date :",on_change=enter_end_date)

    reliance_stock_market_data=yf.download(tickers="RELIANCE.NS",start=start_date,end=end_date)

    if reliance_stock_market_data is not None and st.session_state.start and st.session_state.end:

        st.success("Data Successfully loaded from "+start_date.strftime("%Y-%m-%d")+" to "+end_date.strftime("%Y-%m-%d"))

        st.dataframe(reliance_stock_market_data,width=800)

    input_details=dict(Data=reliance_stock_market_data,Start=start_date,End=end_date)
    
    return input_details


def main():

    np.random.seed(1234)
    rn.seed(1234)
    tf.random.set_seed(1234)
    os.environ["TF_DETERMINISTIC_OPS"]="1"

    activities=["<select>","EDA","Modelling"]
    option=st.sidebar.selectbox("Select Phase",activities)


    if option == "EDA":

        st.write("## Exploratory Data Analysis")

        input_details=display_input_data()
        reliance_stock_market_data=input_details["Data"]

        if st.sidebar.checkbox("Display Shape"):

            st.write("## Shape of Data : ",reliance_stock_market_data.shape)

        if st.sidebar.checkbox("Display Column Names"):

            st.write("## Columns of Data : ",reliance_stock_market_data.columns.tolist())

        if st.sidebar.checkbox("Display Multiple columns and data"):

            st.write("## Select Column Names")
            selected_columns=st.multiselect("",reliance_stock_market_data.columns)
            df1=reliance_stock_market_data[selected_columns]
            st.dataframe(df1)

        if st.sidebar.checkbox("Select Feature to be predicted"):  

            st.write("## Select feature")  
            cols=reliance_stock_market_data.columns.tolist()

            column_name=st.selectbox("",cols)
            
            selected_column=reliance_stock_market_data[column_name]
            st.dataframe(selected_column)

            st.write("## Visualization of Feature")

            st.write("## Moving Average")

            moving_average_100=selected_column.rolling(100).mean()

            moving_average_200=selected_column.rolling(200).mean()

            Moving_Average_df=pd.DataFrame({column_name+" Price":selected_column,"Moving_Average_100":moving_average_100,"Moving_Average_200":moving_average_200})

            fig=px.line(data_frame=Moving_Average_df,
                        x=Moving_Average_df.index,
                        y=Moving_Average_df.columns,
                        title="Moving Average",
                        labels={"Date":"Date","value":column_name+" Price","variable":"Feature"},
                        width=800,height=600)

            st.plotly_chart(fig)

            st.write("## Boxplot")

            fig=px.box(data_frame=Moving_Average_df,
            x=Moving_Average_df.index.year,
            y=Moving_Average_df[column_name+" Price"],
            title="Descriptive Statistics (BOXPLOT)",
            labels={"x":"Year","Price":column_name+" Price"},
            width=800,height=600)
            st.plotly_chart(fig);

    elif option == "Modelling":

        st.write("## Model Building")

        input_details=display_input_data()

        algorithm_name=st.sidebar.selectbox("Select ML Model",("<select>","LSTM"))

        def add_parameter(name_of_algorithm):

                parameter_values=dict()

                if "days" not in st.session_state:

                    st.session_state.days=False

                def enter_days():

                    st.session_state.days=True
                
                if name_of_algorithm=="LSTM":
                    days=st.sidebar.number_input("Days to be predicted",min_value=0,max_value=200,on_change=enter_days)
                    parameter_values["DAYS"]=days

                return parameter_values
        
        parameter_values=add_parameter(algorithm_name)

        if algorithm_name!="<select>" and st.session_state.days:

            def get_classifier(name_of_classifer,params):

                if name_of_classifer == "LSTM" :

                    global scaler

                    closing_amount=input_details["Data"]["Close"]

                    st.write("#### Closing Price of Reliance Stock from "+input_details["Start"].strftime("%Y-%m-%d")+" to "+input_details["End"].strftime("%Y-%m-%d"))

                    st.dataframe(closing_amount)

                    scaler=MinMaxScaler()

                    closing_amount_scaled=scaler.fit_transform(np.array(closing_amount).reshape(-1,1))

                    train_data_length=math.ceil(len(closing_amount_scaled)*0.85)

                    global train_data,test_data

                    train_data=closing_amount_scaled[:train_data_length,:]

                    test_data=closing_amount_scaled[train_data_length:len(closing_amount_scaled),:]

                    def create_dataset(dataset,time_step=1):

                        dataX,dataY=[],[]
                        for i in range(len(dataset)-time_step-1):
                            a=dataset[i:(i+time_step),0]
                            dataX.append(a)
                            dataY.append(dataset[i+time_step,0])
                        return np.array(dataX),np.array(dataY)
                    
                    global time_step

                    time_step=15

                    global x_train,y_train,x_test,y_test

                    x_train,y_train=create_dataset(train_data,time_step)

                    x_test,y_test=create_dataset(test_data,time_step)

                    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)

                    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

                    model=Sequential()
                    model.add(LSTM(units=128,return_sequences=True,input_shape=(x_train.shape[1],1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=64,return_sequences=True))
                    model.add(Dropout(0.3))
                    model.add(LSTM(units=32))
                    model.add(Dense(1))

                    model.compile(optimizer="adam",loss="mean_squared_error")

                else:
                    st.warning("Select your choice of algorithm")

                return model
            
            if st.session_state.days:
            
                model=get_classifier(algorithm_name,parameter_values["DAYS"])

            if "epochs" not in st.session_state:

                st.session_state.epochs=False
            
            def enter_epochs():

                st.session_state.epochs=True

            epochs=st.sidebar.number_input("Epochs",min_value=0,max_value=200,on_change=enter_epochs)

            if st.session_state.epochs:

                model.fit(x=x_train,y=y_train,batch_size=10,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

                pred_train=model.predict(x_train)
                pred_test=model.predict(x_test)

                pred_train=scaler.inverse_transform(pred_train)
                pred_test=scaler.inverse_transform(pred_test)

                y_train_actual=scaler.inverse_transform(y_train.reshape(-1,1))
                y_test_actual=scaler.inverse_transform(y_test.reshape(-1,1))

                rmse_train=math.sqrt(mean_squared_error(y_train_actual,pred_train))
                rmse_test=math.sqrt(mean_squared_error(y_test_actual,pred_test))

                input_data = test_data[len(test_data)-time_step:].reshape(1,-1)
                temp_input_data = input_data.tolist()
                temp_input_data = temp_input_data[0]
                output_data=[]
                i = 0
                prediction_days = parameter_values["DAYS"]

                while(i<prediction_days):
                    
                    if(len(temp_input_data)>time_step):
                        input_data=np.array(temp_input_data[1:])
                        input_data=input_data.reshape(1,-1)
                        input_data=input_data.reshape(1,time_step,1)
                        predicted_data=model.predict(input_data,verbose=0)
                        temp_input_data.extend(predicted_data[0].tolist())
                        temp_input_data=temp_input_data[1:]
                        output_data.extend(predicted_data.tolist())
                        i+=1
                    else:
                        input_data=input_data.reshape(1,time_step,1)
                        predicted_data=model.predict(input_data,verbose=0)
                        temp_input_data.extend(predicted_data[0].tolist())
                        output_data.extend(predicted_data.tolist())
                        i+=1

                future_data=scaler.inverse_transform(output_data).reshape(1,-1).tolist()[0]

                future_days_closing_price=pd.DataFrame({"Predictions":future_data})

                end_date_of_cuurent_data=input_details["Data"].index[len(input_details["Data"])-1]
                days=[]

                for i in range(1,parameter_values["DAYS"]+1):

                    days.append(end_date_of_cuurent_data+datetime.timedelta(days=i))

                future_days_closing_price.index=days

                entered_number=parameter_values["DAYS"]

                if entered_number==1:

                    st.markdown(f"#### Closing Price for the next {entered_number} day predicted")
                else:
                    st.markdown(f"#### Closing Price for the next {entered_number} days predicted")

                st.dataframe(future_days_closing_price)

                future_days_closing_price["Date"]=future_days_closing_price.index

                st.markdown(f"#### Visualization of Predicted Closing Price for the next {entered_number} days")

                fig=px.line(data_frame=future_days_closing_price,x=future_days_closing_price["Date"],y=future_days_closing_price["Predictions"],
                            width=800,height=600)
                
                st.plotly_chart(fig)

                if st.sidebar.checkbox("Display Evaluation Result"):

                    st.write("#### Evaluation Results : ")

                    st.write("#### Training Data")

                    st.write("##### Actual Closing Price of Train Data vs Predicted Closing Price of Train Data")
            
                    fig=plt.figure(figsize=(12,6))
                    plt.plot(y_train_actual,"b",label="Original Price")
                    plt.plot(pred_train,"r",label="Predicted Price")
                    plt.title("Original Closing Price vs Predicted Closing Price")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.legend()
                    st.pyplot(fig)

                    rmse_train=math.sqrt(mean_squared_error(y_train_actual,pred_train))

                    st.write("##### RMSE Value for Training Data is : ",np.round(rmse_train,2))

                    st.write("#### Testing Data")
                            
                    st.write("##### Actual Closing Price of Test Data vs Predicted Closing Price of Test Data")
                
                    fig=plt.figure(figsize=(12,6))
                    plt.plot(y_test_actual,"b",label="Original Price")
                    plt.plot(pred_test,"r",label="Predicted Price")
                    plt.title("Original Closing Price vs Predicted Closing Price")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.legend()
                    st.pyplot(fig)

                    rmse_test=math.sqrt(mean_squared_error(y_test_actual,pred_test))

                    st.write("##### RMSE Value for Testing Data is : ",np.round(rmse_test,2))


                

if __name__ == "__main__":
    main()