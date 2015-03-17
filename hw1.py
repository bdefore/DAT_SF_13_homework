from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#Floats only show two digits for display
pd.set_option('display.precision', 2)

# data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv')
data = pd.read_csv('./titanic.csv')

# since one csv has uppercased column names, standardize to lower
data.columns = [x.lower() for x in data.columns]

# clean data
nan_ages = np.isnan(data.age)
mean_age = data['age'][~nan_ages].mean()
data['age'][nan_ages] = mean_age
data['class'] = data.pclass.map({1: "First", 2: "Second", 3: "Third"})

survivors = data.query('survived == 1')
fatalaties = data.query('survived == 0')
women = data.query('sex == "female"')
men = data.query('sex == "male"')
women_survived = women.query('survived == 1')
men_survived = men.query('survived == 1')
first_class_fatalaties = fatalaties.query('pclass == 1')
second_class_fatalaties = fatalaties.query('pclass == 2')
third_class_fatalaties = fatalaties.query('pclass == 3')
first_class_survivors = survivors.query('pclass == 1')
second_class_survivors = survivors.query('pclass == 2')
third_class_survivors = survivors.query('pclass == 3')
first_class_passengers = data.query('pclass == 1')
second_class_passengers = data.query('pclass == 2')
third_class_passengers = data.query('pclass == 3')

print "1. How many passengers are in our passenger list? From here forward, we'll assume our dataset represents the full passenger list for the Titanic."
print len(data) #1
print "2. What is the overall survival rate?"
print len(survivors) / len(data) #2
print "3. How many male passengers were onboard?"
print len(men) #3
print "4. How many female passengers were onboard?"
print len(women) #4
print "5. What is the overall survival rate of male passengers?"
print len(men_survived) / len(men)
print "6. What is the overall survival rate of female passengers?"
print len(women_survived) / len(women)
print "7. What is the average age of all passengers onboard?"
print data['age'].mean()
print "a. How did you calculate this average age?"
print "Restricting it to people that did not have a NaN age"
print "b. Note that some of the passengers do not have an age value. How did you deal with this? What are some other ways of dealing with this?"
print "Making a mean age per class and assigning that to the NaN aged person based on their class"
print "8. What is the average age of passengers who survived?"
print survivors['age'].mean()
print "9. What is the average age of passengers who did not survive?"
print fatalaties['age'].mean()
print "10. At this (early) point in our analysis, what might you infer about any patterns you are seeing?"
print "Older males were more likely to die than younger females"
print "11. How many passengers are in each of the three classes of service (e.g. First, Second, Third?)"
print data['class'].value_counts()
print "12. What is the survival rate for passengers in each of the three classes of service?"
print "First class:"
print len(first_class_survivors) / len(first_class_passengers)
print "Second class:"
print len(second_class_survivors) / len(second_class_passengers)
print "Third class:"
print len(third_class_survivors) / len(third_class_passengers)
print "13. What else might you conclude?"
print "Rich people are mean"
print "14. Last, if we were to build a predictive model, which features in the data do you think we should include in the model and which can we leave out? Why?"
print "Embarkation point may not indicate where the person is really from"
