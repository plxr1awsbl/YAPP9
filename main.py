import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Opening csv file
path = 'gdp_csv.csv'
data_frame = pd.read_csv(path, sep=',')

#File preparation
countries = np.unique(data_frame['Country Name'])
print('Choose the country you are interested in(Russia is 194, US is 243, UK is 242):')
for i, country in enumerate(countries):
    print(f'{i}) {country}')
choose = int(input())
new_frame = data_frame[data_frame['Country Name'].isin([countries[choose]])]
new_frame = new_frame.drop(['Country Name', 'Country Code'], axis=1)

#Scatter preparation
plt.scatter(x=new_frame['Year'], y=new_frame['Value'])

#Linear Regression
X = new_frame.iloc[:, :-1].values
y = new_frame.iloc[:, 1].values
regressor = LinearRegression()
regressor.fit(X, y)
y_pred = regressor.predict(X)

#Painting regression
plt.plot(X, y_pred, c='red')
plt.show()

#Final prediction
print(f'Enter the year to get predicted GDP in {countries[choose]}')
predict_year = int(input())
predicted_gdp = predict_year * regressor.coef_
print(f'Predicted gdp in {countries[choose]} in {predict_year}: {predicted_gdp}')