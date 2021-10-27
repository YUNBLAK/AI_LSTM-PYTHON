# AI LSTM PYTHON
### AI LSTM PYTHON SOURCE CODES

![SA](https://user-images.githubusercontent.com/87653966/139011900-7d18fd90-e63c-46ce-ba2e-ef919dc0deba.PNG)


This is very simple LSTM model python source code to predict for next 7 days data. You can use this code and apply for your research or study. I kept circulating the Feature value through LSTM, so we could get as many predictive data as we wanted.

###
###

Most LSTM models provided on the Internet produce only one data value or don not provide the number of data we want. Therefore, I created a model that can output as many predicted values as we want.


    def main():
        df = csvloader("NEWCH.csv")

        # NEXT FUTURE n DATA PREDICTION
        # IF DAYS is 7, IT MEANS THAT 7 OUTPUTS WILL COME OUT
        days = 7
        label = "middle"
        batch_size = 120
        arr = LSTMMODEL(df, label, batch_size, days)
        print("NEXT 7 DATA ARE:", arr)
        
### In the main method, there are some variables

    days : It means the number of predicted data we want to get
    label : Feature of datasets
    batch_size : batch size of LSTM(RNN)


The user simply needs to insert a dataset to use this model. This model learns data, predicts and outputs N data from the last datum of the dataset.
