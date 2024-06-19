from datetime import date
import pandas as pd # type: ignore
from sklearn.metrics import accuracy_score,classification_report # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
import pickle as pickle

def create_model(data):
    x= data.drop('diagnosis',axis=1)
    y= data['diagnosis']
    
   
    scaler = StandardScaler()
    x=scaler.fit_transform(x)
    
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    model = LogisticRegression()
    
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    print('Accuracy of our model:',accuracy_score(y_test,y_pred))
    print('Classification report: \n',classification_report(y_test,y_pred)) 
    
    return model,scaler

    
def get_clean_data():
   
    data =pd.read_csv('data/data.csv')
    data =data.drop({'Unnamed: 32','id'},axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    
    return data

def main():
   data = get_clean_data()
   
   model, scaler =create_model(data)
   with open('model/model.pkl','wb') as file:
       pickle.dump(model,file)
   with open('model/scaler.pkl','wb') as file:
       pickle.dump(scaler,file)
             
    
main()    