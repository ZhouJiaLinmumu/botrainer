import argparse
import ipaddress
import json
import os
import pandas as pd
import sys
sys.path.append("..")
import user_agent

import config
import file
from lstm_util import *
import numpy as np
import traceback
import util

def ua_to_float(s):
    if s == 'Other' or s == 'Spider':
        return 1.0
    else:
        return 0.0

def encode_uri(uri, verbose=False):
    encode_uri_size = 15
    expt = None
    tokens = uri.strip('/').split('/')

    if len(tokens) > encode_uri_size:
        tokens[encode_uri_size - 1] = '/'.join(tokens[encode_uri_size - 1:])
        tokens = tokens[:encode_uri_size]
        expt = Exception('encode_uri error: parts of uri: %s large than %d' % (uri, encode_uri_size))

    if verbose:
        if expt:
            util.print_out('encode_uri: %s' % tokens, 1)
            print(expt)
            # time.sleep(1)
        else:
            util.print_out('encode_uri: %s' % tokens)

    for i in range(len(tokens)):
        tokens[i] = util.str_to_float(tokens[i])

    if len(tokens) <= encode_uri_size:
        tokens += [0.0] * (encode_uri_size - len(tokens))

    return tokens

def encode_web(web):
    return [1] if web == "web" else [0]

def encode_type(type):
    ## "Page.SpartanOneBox.AS.Suggestions"
    return [len(type.split("."))]
    

def encode_client_ip(client_ip):
    i = int(ipaddress.IPv4Address(client_ip)) / 4294967295.0
    
    return [i]

def encode_status(status):
    ## /QF_KEYSTROKE_VIRTUAL_URL
    return 

def encode_user_agent(user_agent, verbose=False):
    browser, os, device = parse_useragent(user_agent)
    if verbose:
        print(browser, os, device)

    return [ua_to_float(browser), ua_to_float(os), ua_to_float(device)]
    # return [util.str_to_float(browser), util.str_to_float(os), util.str_to_float(device)]

def encode_host(host, verbose=False):
    encode_host_size = 12
    i = host.find(':')
    if i != -1:
        host = host[:i]

    tokens = host.split('.')

    if util.str_is_ip(host):
        tokens[0] = float(tokens[0])
        tokens[1] = float(tokens[1])
        tokens[2] = float(tokens[2])
        tokens[3] = float(tokens[3])
        tokens += [0.0] * encode_host_size
        if verbose:
            print('encode_host: %s' % tokens)
        return tokens

    tokens = tokens[::-1]

    if len(tokens[0]) > 2:
        tokens.insert(0, '')

    if verbose:
        print('encode_host: %s' % ([0.0] * 4 + tokens))

    for i in range(len(tokens)):
        tokens[i] = util.str_to_float(tokens[i])
    if len(tokens) <= encode_host_size:
        tokens += [0.0] * (encode_host_size - len(tokens))
    else:
        raise Exception('encode_host error: parts of host: %s large than %d' % (host, encode_host_size))

    return [0.0] * 4 + tokens

def encode_query(query):
    ## "qry=sa&cc=US&setlang=en-US&cp=2&cvid=c0f98cdd3..."
    print('encode_query: %s' % query)
    return query



def encode_row(row, verbose=False):
    web = encode_web(row["web"]) #1
    client_ip = encode_client_ip(row["ip"]) ## 1
    typ = encode_type(row["type"])  ## 1
    # user_agent = encode_user_agent(row["agent"]) ## 3
    # query = encode_query(row["query"]
    res = web + client_ip + typ + \
        [int(row["b1"])] + [int(row["b2"])] + \
        [int(row["b4"])] + [int(row["b5"])]

    return res

encoding_vector_size = 7

def get_bing_enumerate_option(target_file):
    option_dic = {
        "type": {},
        "status": {}
    }
    dire = config.STREAM_DIR + "bing-en-us"
    
    count = 0
    for parent_dir in os.listdir(os.path.join(dire)):
        print(parent_dir)
        for filename in os.listdir(os.path.join(dire, parent_dir)):
            try:
                raw_data = pd.read_csv(\
                    os.path.join(dire, parent_dir, filename), header=None)
                count += 1
            except:
                continue
            raw_data.columns = ["time", "id", "web", "type", 'b1', 'b2', 'b3', \
                'b4', 'b5', '5', '6', '7', '8', 'ip', '9', '10', 'status', 'agent', \
                'a', 'b', 'c', 'd', 'e', 'f', 'host', 'query', 'i', 'j', 'k', 'l', 'm', 'n', 'o']

            for _, row_raw_data in raw_data.iterrows():
                if option_dic["type"].get(row_raw_data["type"]) is None:
                    option_dic["type"][row_raw_data["type"]] = 0
                option_dic["type"][row_raw_data["type"]] += 1

                if option_dic["status"].get(row_raw_data["status"]) is None:
                    option_dic["status"][row_raw_data["status"]] = 0
                option_dic["status"][row_raw_data["status"]] += 1
            count += 1

            with open(target_file, 'w') as fp:
                fp.write(json.dumps(option_dic))
        
            if count % 1000 == 0:
                print (count)



if __name__ == '__main__':
    get_bing_enumerate_option("xxx.json")