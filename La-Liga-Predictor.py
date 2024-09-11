import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# loads the data
laliga_data = pd.read_csv("laliga2020.csv", encoding='ISO-8859-1') 
# needs encoding because the file has special characters which makes it hard to process

# if the home goals scored are greater than away goals scored, the home team wins
# vice versa for the away team

def preprocess_results(row):

    # splits the score into two strings so it can be stored in the home_goals and away_goals variables
    home_goals, away_goals = map(int, row['Score'].split('-'))

    # compares the home_goals and away_goals to determine the result of the match
    if home_goals > away_goals:
        return 'Home Win' # need to figure out a way to discern actual team name
    elif home_goals < away_goals:
        return 'Away Win' # need to figure out a way to discern actual team name
    else:
        return 'Draw' 

# apply preprocessing to create a new column for match results
laliga_data['Match Result'] = laliga_data.apply(preprocess_results, axis=1)

# converts categorical data into numerical data for model training
encoder = LabelEncoder()
laliga_data['Home Team Encoded'] = encoder.fit_transform(laliga_data['Home Team'])
laliga_data['Away Team Encoded'] = encoder.fit_transform(laliga_data['Away Team'])

# selecting a subset of features to use in the model
features = [
    'Home Team Rating', 'Away Team Rating', 'Home Team Possession %', 
    'Away Team Possession %', 'Home Team Off Target Shots', 'Away Team Off Target Shots',
    'Home Team Encoded', 'Away Team Encoded'
]

# split the data into features and target
x = laliga_data[features]
y = laliga_data['Match Result']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# initialize the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# trains the model
model.fit(X_train, y_train)

# function to predict the outcome of a specific matchup
def predict_outcome(home_team, away_team):

    # check if there is sufficient historical data for the matchup
    matches = laliga_data[((laliga_data['Home Team'] == home_team) & (laliga_data['Away Team'] == away_team)) |
                           ((laliga_data['Home Team'] == away_team) & (laliga_data['Away Team'] == home_team))]
    
    # if there is no historical data, return a message
    if matches.empty:
        return "No sufficient historical data for this matchup."

    # only average the numeric features used in prediction
    avg_features = matches[features].mean().to_dict()
    feature_values = [avg_features[f] for f in features]
    prediction = model.predict([feature_values])
    return prediction[0]

# user input for matchup prediction
home_team = input("Enter the home team name: ")
away_team = input("Enter the away team name: ")
predicted_result = predict_outcome(home_team, away_team)
print(f"The predicted outcome for {home_team} vs {away_team} is: {predicted_result}")

# Predict on the test set
y_pred = model.predict(X_test)

# evalutes the model
accuracy = 100 * accuracy_score(y_test, y_pred) # retrieves the accuracy provided by the model
report = classification_report(y_test, y_pred) # retrieves the classification report provided by the model

# prints results
print(f'Accuracy: {accuracy}%')
print('Classification Report:')
print(report)
