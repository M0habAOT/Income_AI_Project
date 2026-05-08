
   

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import  OneHotEncoder



encoders={}
def clean(df):
  """
  to clean the data from any extra space and addition punctuation
  """
  df.columns=df.columns.str.strip()
  for col in df.select_dtypes(include="object").columns:
      df[col]=df[col].str.strip().str.rstrip(".")
  return df  
  


def detect_outliers(column,df):
    
    Q1=df[column].quantile(0.25);
    Q3=df[column].quantile(0.75);
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=df[(df[column]<lower_bound) | (df[column]>upper_bound)]
    #remove outliers
    df[column]=df[column].clip(lower=lower_bound,upper=upper_bound)

def labeldata(column,df):
   encoders[column]=LabelEncoder() #for each column in has label encoder
   return encoders[column].fit_transform(df[column]) #fit for train so it learns form it and apply it on test
def labeldatatest(column,df):
    return encoders[column].transform(df[column])
onehotencoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False)#sparsr=false so it doesnot remove the zeros 
def onehot_encoding(df,column):
    onehotencoder.fit(df[column])
    encoded_data= onehotencoder.transform(df[column])
    encoded_df=pd.DataFrame(encoded_data,columns=onehotencoder.get_feature_names_out(column),index=df.index)
    return pd.concat([df.drop(columns=column),encoded_df],axis=1)

def onehot_encoding_test(df,column):
    
    encoded_data= onehotencoder.transform(df[column])
    encoded_df=pd.DataFrame(encoded_data,columns=onehotencoder.get_feature_names_out(column),index=df.index)
    return pd.concat([df.drop(columns=column),encoded_df],axis=1)
   
scale=MinMaxScaler() 
#print(train_data.isnull().sum()) nulls 0
#train data 
if __name__ == "__main__":
    processed_data=pd.read_csv("Preprocessing/train_data.csv")
    processed_data_test=pd.read_csv("Preprocessing/test_data.csv")
    processed_data=clean(processed_data)
    print(processed_data.duplicated().sum())
    print(processed_data[processed_data.duplicated(keep=False)]) #keep =false to also get the duplicated rows not only the original
    processed_data.drop_duplicates(inplace=True) #dropped the duplicates

    print(processed_data.info())
    print(processed_data.iloc[:,[0,2,4,10,11,12]]) # the numeric data
    #before the outliers
    print(processed_data["age"])
    print(processed_data["fnlwgt"])
    print(processed_data["education-num"])
    print(processed_data["capital-gain"])
    print(processed_data["capital-loss"])
    print(processed_data["hours-per-week"])

    #####
    detect_outliers("age",processed_data)
    detect_outliers("fnlwgt",processed_data)
    detect_outliers("education-num",processed_data)
    detect_outliers("capital-gain",processed_data)
    detect_outliers("capital-loss",processed_data)
    detect_outliers("hours-per-week",processed_data)

    #after removing outliers
    print(processed_data["age"])
    print(processed_data["fnlwgt"])
    print(processed_data["education-num"])
    print(processed_data["capital-gain"])
    print(processed_data["capital-loss"])
    print(processed_data["hours-per-week"])



    ##encoding data

    print(processed_data.head())
    processed_data["sex"]=labeldata("sex",processed_data)  #1 for male and 0 for female
    processed_data["Income"]=labeldata("Income",processed_data) # 0 for <=50 and 1 >=50
    print(processed_data.head())  
    processed_data=onehot_encoding(processed_data,["workclass","education","marital-status","occupation","relationship","race","native-country"])
    print(processed_data.head())
        


    #scale the train data

    print(processed_data.head())
    processed_data[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]]=scale.fit_transform(processed_data[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]])
    print(processed_data.head())




    ##for the test data
    #print(processed_data_test.isnull().sum()) no nulls

    processed_data_test=clean(processed_data_test)
    print(processed_data_test.duplicated().sum())
    print(processed_data_test[processed_data_test.duplicated(keep=False)]) #keep =false to also get the duplicated rows not only the original
    processed_data_test.drop_duplicates(inplace=True) #dropped the duplicates

    print(processed_data_test["age"])
    print(processed_data_test["fnlwgt"])
    print(processed_data_test["education-num"])
    print(processed_data_test["capital-gain"])
    print(processed_data_test["capital-loss"])
    print(processed_data_test["hours-per-week"])

    #####
    detect_outliers("age",processed_data_test)
    detect_outliers("fnlwgt",processed_data_test)
    detect_outliers("education-num",processed_data_test)
    detect_outliers("capital-gain",processed_data_test)
    detect_outliers("capital-loss",processed_data_test)
    detect_outliers("hours-per-week",processed_data_test)

    #after removing outliers
    print(processed_data_test["age"])
    print(processed_data_test["fnlwgt"])
    print(processed_data_test["education-num"])
    print(processed_data_test["capital-gain"])
    print(processed_data_test["capital-loss"])
    print(processed_data_test["hours-per-week"])




    ##encoding data

    print(processed_data_test.head())
    processed_data_test["sex"]=labeldatatest("sex",processed_data_test)  #1 for male and 0 for female
    processed_data_test["Income"]=labeldatatest("Income",processed_data_test) # 0 for <=50 and 1 >=50
    print(processed_data_test.head())  
    processed_data_test=onehot_encoding_test(processed_data_test,["workclass","education","marital-status","occupation","relationship","race","native-country"])
    print(processed_data_test.head())
        

    #scaling testdata

    processed_data_test[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]]=scale.transform(processed_data_test[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]])
    print(processed_data_test.head())  #trasnform without fit it learned the min and max form train data so it willnot memorize test data parameters

    processed_data.to_csv("Preprocessing/processed_train_data.csv", index=False)
    processed_data_test.to_csv("Preprocessing/processed_test_data.csv", index=False)
  




    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(onehotencoder, "onehotencoder.pkl")
    joblib.dump(scale, "scale.pkl")

