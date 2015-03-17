from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#Floats only show two digits for display
pd.set_option('display.precision', 2)

# titanic_data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv')
titanic_data = pd.read_csv('./titanic.csv')
titanic_data.columns = [x.lower() for x in titanic_data.columns]

# print type(titanic_data)
# print titanic_data.head()

all_people = []
survivors = []
fatalaties = []
women = []
men = []
women_survived = []
men_survived = []
total_survivor_age = 0
total_fatality_age = 0

# clean data
nan_ages = np.isnan(titanic_data.age)
mean_age = titanic_data['age'][~nan_ages].mean()
titanic_data['age'][nan_ages] = mean_age

for index, person in titanic_data.iterrows():
  all_people.insert(0, person);
  if person.sex == 'male':
    men.insert(0, person)
    if person.survived == 1:
      men_survived.insert(0, person)
  if person.sex == 'female':
    women.insert(0, person)
    if person.survived == 1:
      women_survived.insert(0, person)
  if person.survived == 1:
    survivors.insert(0, person)
    total_survivor_age += person.age
  else:
    fatalaties.insert(0, person)
    total_fatality_age += person.age

titanic_data["class"] = titanic_data.pclass.map({1: "First", 2: "Second", 3: "Third"})

first_class_fatalaties = titanic_data[(titanic_data['survived'] == 0) & (titanic_data['pclass'] == 1)]
first_class_survivors = titanic_data[(titanic_data['survived'] == 1) & (titanic_data['pclass'] == 1)]
second_class_fatalaties = titanic_data[(titanic_data['survived'] == 0) & (titanic_data['pclass'] == 2)]
second_class_survivors = titanic_data[(titanic_data['survived'] == 1) & (titanic_data['pclass'] == 2)]
third_class_fatalaties = titanic_data[(titanic_data['survived'] == 0) & (titanic_data['pclass'] == 3)]
third_class_survivors = titanic_data[(titanic_data['survived'] == 1) & (titanic_data['pclass'] == 3)]
first_class_passengers = titanic_data[(titanic_data['pclass'] == 1)]
second_class_passengers = titanic_data[(titanic_data['pclass'] == 2)]
third_class_passengers = titanic_data[(titanic_data['pclass'] == 3)]

print "1. How many passengers are in our passenger list? From here forward, we'll assume our dataset represents the full passenger list for the Titanic."
print len(all_people) #1
print "2. What is the overall survival rate?"
print len(survivors) / len(all_people) #2
print "3. How many male passengers were onboard?"
print len(men) #3
print "4. How many female passengers were onboard?"
print len(women) #4
print "5. What is the overall survival rate of male passengers?"
print len(men_survived) / len(men) #5
print "6. What is the overall survival rate of female passengers?"
print len(women_survived) / len(women)
print "7. What is the average age of all passengers onboard?"
print titanic_data['age'].mean()
print "a. How did you calculate this average age?"
print "Restricting it to people that did not have a NaN age"
print "b. Note that some of the passengers do not have an age value. How did you deal with this? What are some other ways of dealing with this?"
print "Making a mean age per class and assigning that to the NaN aged person based on their class"
print "8. What is the average age of passengers who survived?"
print total_survivor_age / len(survivors)
print "9. What is the average age of passengers who did not survive?"
print total_fatality_age / len(survivors)
print "10. At this (early) point in our analysis, what might you infer about any patterns you are seeing?"
print "Older males were more likely to die than younger females"
print "11. How many passengers are in each of the three classes of service (e.g. First, Second, Third?)"
print titanic_data["class"].value_counts()
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

