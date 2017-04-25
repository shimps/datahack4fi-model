import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel

all_data = pd.read_csv('updated_model_sorted.csv', low_memory=False)

#undersampling to help with the large imbalance
raw_data= all_data.iloc[:3500].copy()


chosen_columns = ['RU','Province','District','HH_size','Adults','A11','Q1_2','Q1_3',
                  'Q6_2_4_c','Q6_2_5_c','Q10_1_9','Banked_c','Saving_Strand','Insurance_Strand','i2i_Education','i2i_Income_Sources']

continuous_columns = ['HH_size','Adults','Q1_2']

final_score = 0

#huge imbalance means, there might be too few training positives.
while final_score < 0.75:
    categorical_columns = [key for key in chosen_columns if key not in continuous_columns]

    

    processed_data = pd.DataFrame()
    for key in chosen_columns:
        processed_data[key] = raw_data[key]



    #Helps to split train test 80% 20%
    msk = np.random.rand(len(processed_data)) < 0.8

    target_train = processed_data[msk]['Insurance_Strand'].values
    target_test = processed_data[~msk]['Insurance_Strand'].values

    del processed_data['Insurance_Strand']
    categorical_columns.remove('Insurance_Strand')

    trans_data = pd.get_dummies(processed_data, columns= categorical_columns)
    #processed_data[categorical_columns] = processed_data[categorical_columns].apply(LabelEncoder().fit_transform)
    #trans_data = processed_data

    train_data = trans_data[msk]
    test_data = trans_data[~msk]


    model = ExtraTreesClassifier(n_estimators = 250)
    model.fit(train_data.values, target_train)
    predictions = model.predict(test_data)
    score = f1_score(target_test,predictions,average=None)
    print score
    final_score = score[0]
    importance_matrix = list(model.feature_importances_)
    full_values = []
    for i in range(0,len(importance_matrix)):
        full_values.append((i,importance_matrix[i]))

    full_values.sort(key=lambda x: x[1],reverse = True)

    #print full_values

    def find(value):
        index = importance_matrix.index(value)
        print "Column: %s "%(trans_data.columns[index])

    def find_index(value,imp):
        print "Column: %s Imp: %s"%(trans_data.columns[value],imp)

    def execute():
        for value in full_values:
            find_index(value[0],value[1])

