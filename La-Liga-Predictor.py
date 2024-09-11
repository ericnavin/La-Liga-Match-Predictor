import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# loads the data
laliga_data = pd.read_csv("laliga2020.csv", encoding='ISO-8859-1') 
# needs encoding because the file has special characters which makes it hard to process

# Preprocess the Score to determine Home Win, Draw or Away Win
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

# Apply preprocessing to create a new column for match results
laliga_data['Match Result'] = laliga_data.apply(preprocess_results, axis=1)

# Convert categorical data into numerical data for model training
encoder = LabelEncoder()
laliga_data['Home Team Encoded'] = encoder.fit_transform(laliga_data['Home Team'])
laliga_data['Away Team Encoded'] = encoder.fit_transform(laliga_data['Away Team'])

# Selecting a few relevant features for simplicity
features = [
    'Home Team Rating', 'Away Team Rating', 'Home Team Possession %', 
    'Away Team Possession %', 'Home Team Off Target Shots', 'Away Team Off Target Shots',
    'Home Team Encoded', 'Away Team Encoded'
]

x = laliga_data[features]
y = laliga_data['Match Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)
