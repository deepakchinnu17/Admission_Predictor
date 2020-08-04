import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r"C:\Users\DELL\PycharmProjects\US_PREDICT\MINIPROJECT_DATA.csv")
df=df.rename(columns={'Serial No.':'no','GRE Score':'gre','TOEFL Score':'toefl','University Rating':'rating','SOP':'sop','LOR ':'lor',
                           'CGPA':'gpa','Research':'research','Chance of Admit ':'chance'})
df.drop(['no'],axis=1,inplace=True)
var=df.columns.values.tolist()
y=df['chance']
Z=['no', 'chance']
x=[i for i in var if i not in Z]
x=df[x]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
#NORMALIZATION MAY NOT BE NEEDED

xs=MinMaxScaler()
x_train[x_train.columns] = xs.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = xs.transform(x_test[x_test.columns])

#END

cy_train=[1 if chance > 0.83 else 0 for chance in y_train]
cy_train=np.array(cy_train)

cy_test=[1 if chance > 0.83 else 0 for chance in y_test]
cy_test=np.array(cy_test)

lr = LogisticRegression()
lr.fit(x_train, cy_train)
'''
inp1=330
inp2=110
inp3=3
inp4=4
inp5=4
inp6=8.7
inp7=0
l=[inp1, inp2, inp3, inp4, inp5, inp6, inp7]
#print(lr.predict(l))
new_data = [(330,115,5,5,4,8.5,1), (280,80,3,4,4,8.7,1),(340,120,5,5,5,8.7,1)]
#Convert to numpy array
new_array = np.asarray(new_data)
#Output Labels
labels=["reject","admit"]
#Let's make some kickass predictions
prediction=lr.predict(new_array)
#Get number of test cases used
print(prediction)
print(x_test)
no_of_test_cases, cols = new_array.shape
for i in range(no_of_test_cases):
 print("Status of Student with GRE scores = {}, GPA grade = {}, Rank = {} will be ----- {}".format(new_data[i][0],new_data[i][1],new_data[i][2], labels[int(prediction[i])]))
'''
file=open('US_prediction.pkl', 'wb')
pickle.dump(lr, file)

