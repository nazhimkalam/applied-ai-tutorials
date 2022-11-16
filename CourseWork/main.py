loadedModel = joblib.load('lung-cancer-pred-model.pkl')

validationDataset = pd.read_csv('./validation.csv')
validationDataset.head()

# loop through the validation dataset and make predictions
for index, row in validationDataset.iterrows():
    # make sure the input data for the model is in this format
    # [[1, 21, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2]]

    # get the data from the row
    data = row[0:15].values.tolist()

    # make a prediction
    prediction = loadedModel.predict([data])

    # actual value
    actual = row[16]


    # print the prediction and actual value
    print('Prediction: ', prediction, 'Actual: ', actual)
