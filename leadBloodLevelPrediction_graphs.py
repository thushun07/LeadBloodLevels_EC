import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

n = 3000

# simulate data set to run the model

def lead_paint(year): 
    """
    returns the probability that there is lead paint 
    given how many years before 2025 the house was built
    """

    if year<48:
        return 0
    elif year<65:
        return 0.24
    elif year<85:
        return 0.69
    else:
        return 0.87
    
def soil_lead_exposure(soil_ppm):
    """
    Returns the probability of lead exposure depending on the lead levels
    of the soil in the house's yard
    """
    if soil_ppm <= 100: #baseline meaning no exposure
        return 0.0 
    elif soil_ppm <= 1000:
        return 1 #returns the prob of lead exposure (1) * weight of impact (1)
    elif soil_ppm <= 2000:
        return 2 #returns the prob of lead exposure (1) * weight of impact (2)
    else:
        return 2

house_age = np.random.uniform(0,120,n) #house age varies continuously
leaded_paint = np.array([np.random.binomial(1, lead_paint(age)) for age in house_age]) #returns 1 or 0 depending on if there is leaded paint and stores it in an array
poverty_level = np.random.binomial(1, 0.167, n) #returns 1 or 0 depending on whether the house is below the poverty line
soil_ppm = np.random.choice([100, 1000, 2000], size=n) #chooses a random soil level
soil = np.array([soil_lead_exposure(s) for s in soil_ppm]) #calculates the probability of lead exposure


# hidden logistic model
z = -1 + leaded_paint + 2*poverty_level + soil
p = 1/(1+np.exp(-z))

# sample outcomes
elevated = np.random.binomial(1,p)

data = pd.DataFrame({
    "house_age": house_age,
    "lead_paint": leaded_paint,
    "poverty_level": poverty_level,
    "soil_ppm": soil_ppm,
    "soil_prob": soil,
    "elevated_lead": elevated
})

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#now we want to fit a logistic regression model to estimate probabilities
# Features
X = data[["house_age", "lead_paint", "poverty_level", "soil_prob"]].values
y = data["elevated_lead"].values

# add intercept column
X = np.c_[np.ones(X.shape[0]), X]

weights = np.zeros(X.shape[1])

#train using gradient ascent
learning_rate = 0.0001
iterations = 5000

for i in range(iterations):
    
    # linear combination
    z = np.dot(X, weights)
    
    # predicted probabilities
    predictions = sigmoid(z)
    
    # gradient of log loss
    gradient = np.dot(X.T, (predictions - y)) / len(y)
    
    # update weights
    weights -= learning_rate * gradient

#predict probabilities
def predict_prob(X_input):
    
    X_input = np.array(X_input)
    X_input = np.c_[np.ones(X_input.shape[0]), X_input]
    
    z = np.dot(X_input, weights)
    return sigmoid(z)

# Estimated probabilities
data["predicted_prob"] = predict_prob(
    data[["house_age","lead_paint","poverty_level","soil_prob"]].values
)


#visualize the data using a histogram
plt.hist(data["predicted_prob"], bins=10, alpha=0.7)
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Houses")
plt.show()

ages = np.linspace(0,120,100)

preds = [
    predict_prob([[age, lead_paint(age), 0, 2.25]])[0]
    for age in ages
]

plt.plot(ages, preds)
plt.xlabel("House Age")
plt.ylabel("Predicted Probability")
plt.title("Predicted Lead Exposure Risk vs House Age")
plt.show()