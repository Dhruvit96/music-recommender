# music-recommender
Using DecisionTree algorithm giving user music genre based on their age and sex.

## Idea of persisting models is that larger datasets may take minutes or hours to train a model so we don't wanna build and train a model everytime the program runs. Ideal practice is to every once in a while build and train a model and save it for later to make predictions in the program.

# first step
Importing the data
music_data = pd.read_csv('music.csv')
# second step
Preparing data - X (input set) and y (output set). 
Note: No cleaning is required since there are no null values or duplicates in the dataset
# third step
Spliting input and output dataset into input and output training and testing dataset (usually keep 20% of dataset from testing while 80% fro training the model). Note: More data we give the model to train the more accurate predictions it'll make.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# forth step
Create a model using a machine learning algorithm like DecisionTree or Neural-Network etc. The choice of algorithm depends on its performance and accuracy
model = DecisionTreeClassifier()
Training model with input and output training set
model.fit(X_train, y_train)

# After training the model, save the model as a .joblib file so we don't have to train model every time.

# final step
Test
