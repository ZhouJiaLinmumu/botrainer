import ipaddress
import math
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset
from user_agents import parse
import xgboost


from classification_setting import train_file_id, test_file_id
import util

methods = {
    "SVM": svm.SVC(gamma=0.01, verbose=True),
    "XGBoost": xgboost.XGBRegressor(),
    "MLP": MLPClassifier(),
    "Tree": DecisionTreeClassifier(max_depth=5),
    "RF": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "ADA": AdaBoostClassifier()
}



def ua_to_float(s):
    if s == 'Other' or s == 'Spider':
        return 1.0
    else:
        return 0.0

def parse_useragent(s):
    user_agent = parse(s)
    return user_agent.browser.family, user_agent.os.family, user_agent.device.family

def agent_encoding(user_agent, verbose=True):
    browser, os, device = parse_useragent(user_agent)

    return [ua_to_float(browser), ua_to_float(os), ua_to_float(device)]

def count_encoding(count):
    return [count]

def client_ip_encoding(client_ip):
    i = int(ipaddress.IPv4Address(client_ip)) / 4294967295.0
    return [i]


def feature_encoding(row):
    client_ip = client_ip_encoding(row["client_ip"])
    agents = agent_encoding(row["agents"])
    count = count_encoding( row["counts"] )
    return agents + count + client_ip

def parse_metadata(file_id, encoding_method):
    tags = pd.read_csv(file_id + "_session.tag.csv").iloc[:,1].values
    meta_df = pd.read_csv(file_id + "_session.metadata.csv")
    if len(meta_df.columns) == 3:
        meta_df.columns = ["client_ip", "host", "agents"]
    else:
        meta_df.columns = ["client_ip", "host", "agents", "percentage"]

    meta_df["counts"] = pd.read_csv(file_id + "_session.count.csv").values[:, 1]
    dataset = []
    for index, row_raw_data in meta_df.iterrows():
        try:
            encoded = encoding_method(row_raw_data)
            dataset.append(encoded)
        except (ipaddress.AddressValueError, TypeError):
            tags[index] = -2
        if index % 10000 == 0:
            print (index)
    tags = list(filter(lambda x: x != -2, tags))
    return dataset, tags

class SVMDataset(object):
    def __init__(self, file_id, encoding_method=feature_encoding):
        self.dataset = parse_metadata(file_id, encoding_method)

def main(class_name_list):
    
    ## Fitting
    train_dataset = SVMDataset(train_file_id).dataset
    train_features = train_dataset[0] 
    train_labels = train_dataset[1]

    ## Predict
    test_dataset = SVMDataset(test_file_id).dataset
    test_features = test_dataset[0]
    test_labels = test_dataset[1]

    for class_name in class_name_list:
        clf = methods[class_name]
        clf.fit(train_features, train_labels)
    
        test_pred_raw = clf.predict(test_features)
        test_pred = list(map(lambda x: int(x > 0.5), test_pred_raw))
        tn, fp, fn, tp = confusion_matrix(test_labels, test_pred).ravel()
        print(class_name)
        print ("TN {0}, FP {1}, FN {2}, TP{3}".format(str(tn), str(fp), str(fn), str(tp)))
        print(classification_report(test_labels, test_pred, target_names=["bot", "nonbot"]))
        print()
        print()

if __name__ == '__main__':
    # main("ADA")
    # main("MLP")
    main(["XGBoost", "ADA", "MLP", "Tree", "RF", "SVM"])
