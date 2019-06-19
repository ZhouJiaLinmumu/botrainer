# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ipaddress

import config
import file
from label import tag
import util

from enum import Enum


class Column(Enum):
    Timestamp = 0
    PreciseTimeStamp = 1
    ServicePrefix = 2
    Region = 3
    GatewayId = 4
    Tenant = 5
    Role = 6
    RoleInstance = 7
    ResourceId = 8
    operationName = 9
    time = 10
    category = 11
    instanceId = 12
    clientIp = 13
    clientPort = 14
    httpMethod = 15
    requestUri = 16
    userAgent = 17
    httpStatus = 18
    httpVersion = 19
    receivedBytes = 20
    sentBytes = 21
    timeTaken = 22
    sslEnabled = 23
    host = 24
    requestQuery = 25
    cacheHit = 26
    serverRouted = 27
    logId = 28
    serverStatus = 29
    rowKey = 30
    sourceEvent = 31
    sourceMoniker = 32


method_map = {'GET': 0,
              'POST': 1,
              'PUT': 2,
              'DELETE': 3,
              'OPTIONS': 4,
              'PATCH': 5,
              'HEAD': 6,
              'PROPFIND': 7,
              'REPORT': 8}


example_row = ['2018-10-11T02:33:30.0000000Z', '2018-10-11T02:33:30.0000000Z', 'wavnet', 'South Central US', '4dc1135e-7f9f-44a7-8909-a700761c92a4', '568f3a8f98a0475fa24a6351b63433c7', 'ApplicationGatewayRole', 'ApplicationGatewayRole_IN_0', '/SUBSCRIPTIONS/111', 'ApplicationGatewayAccess', '2018-10-11T02:32:01.0000000Z', 'ApplicationGatewayAccessLog', 'ApplicationGatewayRole_IN_0',
               '1.2.3.4', '5432', 'GET', 'abc/page.html', 'Mozilla/5.0', '200', 'HTTP/1.1', '123', '123', '123', 'on', 'example.com', 'abc=def']


def get_encoded_size():
    vector = encode_row(example_row, verbose=False)
    return len(vector)


def encode_row(row, verbose=True):
    # timestamp = row[Column.Timestamp.value]
    # tenant = row[Column.Tenant.value]
    method = row[Column.httpMethod.value]
    uri = row[Column.requestUri.value]
    user_agent = row[Column.userAgent.value]
    status = row[Column.httpStatus.value]
    host = row[Column.host.value]
    client_ip = row[Column.clientIp.value]
    # client_port = row[Column.clientPort.value]
    # query = row[Column.requestQuery.value]

    method = encode_method(method)
    uri = encode_uri(uri, verbose)
    user_agent = encode_user_agent(user_agent, verbose)
    status = encode_status(status)
    host = encode_host(host, verbose)
    client_ip = encode_client_ip(client_ip)
    # query = encode_query(query)

    if verbose:
        print("method: " + method.__str__())
        print("uri: " + uri.__str__())
        # print("user_agent: " + user_agent.__str__())
        print("status: " + status.__str__())
        print("host: " + host.__str__())
        print("client_ip: " + client_ip.__str__())

    res = method + uri + user_agent + status + host + client_ip
    # res = method + uri + status + host + client_ip
    return res


def encode_method(method):
    if method in method_map:
        return [method_map[method] / 10.0]
    else:
        return [1.0]


encode_uri_size = 15


def encode_uri(uri, verbose=True):
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


def ua_to_float(s):
    if s == 'Other' or s == 'Spider':
        return 1.0
    else:
        return 0.0


def encode_user_agent(user_agent, verbose=True):
    browser, os, device = tag.parse_useragent(user_agent)
    if verbose:
        print(browser, os, device)

    return [ua_to_float(browser), ua_to_float(os), ua_to_float(device)]
    # return [util.str_to_float(browser), util.str_to_float(os), util.str_to_float(device)]


def encode_status(status):
    return [int(status) / 600.0]


encode_host_size = 12


def encode_host(host, verbose=True):
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


def encode_client_ip(client_ip):
    i = int(ipaddress.IPv4Address(client_ip)) / 4294967295.0
    return [i]

    # tokens = host.split('.')
    # res = [int(tokens[0]) / 256.0, int(tokens[1]) / 256.0, int(tokens[2]) / 256.0, int(tokens[3]) / 256.0]
    # # print(res)
    # return res


def encode_query(query):
    print('encode_query: %s' % query)
    return query


def strip_user_agent(user_agent):
    start = user_agent.find('(')
    if start != -1:
        return user_agent[:start]
    else:
        return user_agent


def test_print_rows():
    for row in file.yield_data_row(config.LOG_EXAMPLE_SMALL):
        print('**********************************************')
        print(' || '.join(row))

        numbers = encode_row(row)
        print(numbers)


def test_encode_row():
    vector = encode_row(example_row)
    return len(vector)


def test_field(i, encode_func):
    existed = {}
    for row in file.yield_data_row(config.LOG_EXAMPLE):
        field = row[i]
        if field not in existed:
            val = encode_func(field)
            print('%s --> %s' % (field, val))
            existed[field] = 1
        else:
            # print('')
            pass


def test_user_agent():
    existed = {}
    for row in file.yield_data_row(config.LOG_EXAMPLE):
        user_agent = strip_user_agent(row[Column.userAgent.value])
        if user_agent not in existed:
            encode_user_agent(user_agent)
            existed[user_agent] = 1


if __name__ == '__main__':
    # test_field(Column.requestUri.value, encode_uri)
    # test_field(Column.host.value, encode_host)

    # test_field(Column.httpMethod.value, encode_method)
    # test_field(Column.httpStatus.value, encode_status)
    # test_field(Column.clientIp.value, encode_client_ip)

    # test_field(Column.requestQuery.value, encode_query)

    # test_user_agent()

    # test_encode_row()

    # print(get_encoded_size())

    # test_print_rows()

    # encode_uri('/Relativity/Identity/connect/authorize')
    # encode_uri('/')
    # encode_uri('/relativity/')

    print(encode_user_agent('Mozilla/5.0+(Windows+NT+10.0;+Win64;+x64)+AppleWebKit/537.36+(KHTML,+like+Gecko)+Chrome/68.0.3440.106+Safari/537.36'))
    print(encode_user_agent('LogicMonitor+SiteMonitor/1.0'))

    # print(encode_client_ip('127.0.0.1'))
    # print(encode_client_ip('127.0.0.2'))
    # print(encode_client_ip('127.0.1.1'))
    # print(encode_client_ip('127.1.0.1'))
    # print(encode_client_ip('93.159.171.43'))
    # print(encode_client_ip('255.255.255.255'))

    # encode_host('40.113.216.110:80')
    # encode_host('tsemerge.relativity.one')
