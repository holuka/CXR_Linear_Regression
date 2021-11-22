import math
import csv
import cv2
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression

directory_address_list = [["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/train/no/", 15]];
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/train/yes/",  15]);
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/test/no/", 10]);
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/test/yes/", 10]);
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/evaluate/no/",5]);
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/evaluate/yes/",  5]);
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/test-train/no/",25]);
directory_address_list.append(["/Users/statcscuser/Downloads/OneDrive/HINF 491/Individual Assigment/dataset/test-train/yes/",  25]);

#for every folder in liste of file addresses
#-read files into liste
files_list = list();
directory_name_list = list()
for directory in directory_address_list:


    number_of_files = directory[1];
    directory_name = directory[0][len(directory[0])-14:len(directory[0])];
    directory_name_list.append([directory_name,number_of_files])

    number_of_files = directory[1];
    for i in (range(number_of_files)):
        filename = "";
        if (directory_name == "aset/train/no/"):
            filename = "image"
        if (directory_name == "set/train/yes/"):
            filename = "photo"
        if (directory_name == "taset/test/no/"):
            filename = "image"
        if (directory_name == "aset/test/yes/"):
            filename = "photo"
        if (directory_name == "t/evaluate/no/"):
            filename = "NORMAL2-"
        if (directory_name == "/evaluate/yes/"):
            filename = "000001-"
        if (directory_name == "test-train/no/"):
            filename = "image"
        if (directory_name == "est-train/yes/"):
            filename = "image"

        file_address = directory[0]+filename+str(i+1)+".jpeg";
        image = cv2.imread(file_address, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (255, 255))
        image = image/255
        file_data_array_row_number,file_data_array_column_number = image.shape
        file_data_array = image
        files_list.append([directory_name,file_data_array,file_data_array_row_number,file_data_array_column_number]);

model_list = list();
#for each file in list of file
#-apply linear model
for  file in files_list:
    file_data_array_row_number = file[3]
    data = np.array(file[1])
    #data = data.reshape(1,-1)
    y = np.arange(file_data_array_row_number)+np.ones((file_data_array_row_number),dtype=np.int16);
    #y = y.reshape(1,-1)
    model = LinearRegression(fit_intercept=False, normalize=True).fit(data,y);
    model_list.append([file[0],model]);  #append directory_name and model to the model_list
#estimated standard error
def column_ese (table):
  row_length, column_length = table.shape
  table_ese = list()
  for i in range(column_length):
    column_ese = math.sqrt(np.var(table[:,i]))
    table_ese.append(column_ese)
  return np.array(table_ese).reshape(1,-1)
#mean
def column_mean (table):
  row_length, column_length = table.shape
  table_mean = list()
  for i in range(column_length):
    column_mean = np.mean(table[:,i])
    table_mean.append(column_mean)
  return np.array(table_mean).reshape(1,-1)
#Confidence interval
def CI(est,ese):
  param_table = np.column_stack((est.reshape(-1,1),ese.reshape(-1,1)))
  row_length, column_length = param_table.shape
  CI_table = list ()
  for i in range(row_length) :
    esti =param_table[i][0]
    esei = param_table[i][1]
    alpha = 0.05
    cv = scipy.stats.t.ppf(alpha/2., row_length-1)
    CI_L = esti + (cv*esei*-1)
    CI_H = esti + (cv*esei)
    CI_esti = [CI_L,CI_H]  #CI_table row i
    CI_table.append(CI_esti)
  return(np.array(CI_table))

def returnBetaEstimateTable(dataset):
    beta_estimates_table = list()
    for file_in_dataset in dataset:  #each model in dataset
        model_in_file = file_in_dataset[1]
        beta_estimate_list_per_file_in_dataset= model_in_file.coef_
        beta_estimates_table.append(beta_estimate_list_per_file_in_dataset)
    return(beta_estimates_table)

train_no_dataset = list ()
train_yes_dataset = list ()
test_no_dataset = list ()
test_yes_dataset = list ()
evaluate_no_dataset = list ()
evaluate_yes_dataset = list ()
test_train_no_dataset = list ()
test_train_yes_dataset = list ()

for model in model_list:
    if model[0] == directory_name_list[0][0]: #if filename = first directory name
        train_no_dataset.append([model[0],model[1],directory_name_list[0][1]]) #append [directory_name, model, number_of_files_in_directory]
    if model[0] == directory_name_list[1][0]:
        train_yes_dataset.append([model[0],model[1],directory_name_list[1][1]])
    if model[0] == directory_name_list[2][0]:
        test_no_dataset.append([model[0],model[1],directory_name_list[2][1]])
    if model[0] == directory_name_list[3][0]:
        test_yes_dataset.append([model[0],model[1],directory_name_list[3][1]])
    if model[0] == directory_name_list[4][0]:
        evaluate_no_dataset.append([model[0],model[1],directory_name_list[4][1]])
    if model[0] == directory_name_list[5][0]:
        evaluate_yes_dataset.append([model[0],model[1],directory_name_list[5][1]])
    if model[0] == directory_name_list[6][0]:
        test_train_no_dataset.append([model[0],model[1],directory_name_list[6][1]])
    if model[0] == directory_name_list[7][0]:
        test_train_yes_dataset.append([model[0],model[1],directory_name_list[7][1]])

#for every model in training set, check if the estimate is within CI of any estimate predicted in the test set
def predict_model(training, test):
    for  i in range(len(training)):
        for j in range(len(test)):
            train_no_model_summary_table = np.array(returnBetaEstimateTable(training)).reshape(training[1][2],-1)
            train_yes_model_summary_table = np.array(returnBetaEstimateTable(test)).reshape(test[1][2],-1)

            # print("train_no_model_summary_table.shape")
            # print(train_no_model_summary_table.shape)

            #obtain column_mean
            train_no_model_summary_table_column_mean = column_mean(train_no_model_summary_table)
            train_yes_model_summary_table_column_mean = column_mean(train_yes_model_summary_table)

            # print("train_no_model_summary_table_column_mean.shape")
            # print(train_no_model_summary_table_column_mean.shape)


            #obtain column_variance
            train_no_model_summary_table_column_ese = column_ese(train_no_model_summary_table)
            train_yes_model_summary_table_column_ese = column_ese(train_yes_model_summary_table)

            # print("train_no_model_summary_table_column_ese.shape")
            # print(train_no_model_summary_table_column_ese.shape)


            #obtain beta_in_model_CI
            train_no_model_summary_table_column_CI = CI(train_no_model_summary_table_column_mean, train_no_model_summary_table_column_ese)
            train_yes_model_summary_table_column_CI = CI(train_yes_model_summary_table_column_mean, train_no_model_summary_table_column_ese)

            # print("train_no_model_summary_table_column_CI")
            # print(train_no_model_summary_table_column_CI.shape)
            # print(train_no_model_summary_table_column_CI.shape)

            #create beta_in_model_summary_table for each dataset with columns [column_mean, column_variance, CI_L , CI_H]
            train_no_beta_in_model_summary_table = np.column_stack((train_no_model_summary_table_column_mean.reshape(-1,1), train_no_model_summary_table_column_ese.reshape(-1,1), train_no_model_summary_table_column_CI[:,0], train_no_model_summary_table_column_CI[:,1]))
            train_yes_beta_in_model_summary_table = np.column_stack((train_yes_model_summary_table_column_mean.reshape(-1,1), train_yes_model_summary_table_column_ese.reshape(-1,1), train_yes_model_summary_table_column_CI[:,0], train_yes_model_summary_table_column_CI[:,1]))

            # print("train_no_beta_in_model_summary_table.shape")
            # print(train_no_beta_in_model_summary_table.shape)



            compare_test_train_No_vs_test_Yes = np.any(train_yes_beta_in_model_summary_table[:,0]>train_no_beta_in_model_summary_table[:,2]) and np.any(train_yes_beta_in_model_summary_table[:,0]<train_no_beta_in_model_summary_table[:,3])
            # print("compare_test_train_No_vs_test_Yes")
            # print(compare_test_train_No_vs_test_Yes)

            if (compare_test_train_No_vs_test_Yes):
                print("\nRESULT:  COVID19 detected.\n")
            else:
                print("\nRESULT:  COVID19 not detected.\n")

print("\nChecking for COVID-19 in COVID-19 positive dataset \n")
predict_model (test_train_yes_dataset,evaluate_yes_dataset)

print("\nChecking for no COVID-19 in COVID-19 positive dataset \n")
predict_model (test_train_yes_dataset,evaluate_yes_dataset)

print("\nChecking for no COVID-19 in COVID-19 negative dataset \n")
predict_model (test_train_yes_dataset,evaluate_no_dataset)

print("\nChecking for COVID-19 in COVID-19 negative dataset \n")
predict_model (test_train_yes_dataset,evaluate_no_dataset)
