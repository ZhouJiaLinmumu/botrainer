import json
import requests
import sys
sys.path.append("../")


import cnn_dataset
import file
import util


 # class RemoteMyDataset(cnn_dataset.MyDataset):
#     def __init__(self, file_id, host='http://localhost', port=8080):
#         super(RemoteMyDataset, self).__init__(file_id)
#         self.host = host
#         self.port = port
#         self.info_cache = {}
#         self.inference_list = file.read_session_guess_file(util.get_session_guess_path(cnn_dataset.test_file_id))

 #     def __getitem__(self, index):
#         if self.info_cache.get(index):
#             print('Cache hit..')
#             return self.info_cache[index]

 #         tensor, tag = super(RemoteMyDataset, self).__getitem__(index)
#         trunk_id, session_id = self.file_map[index]

 #         payload = {
#             'fileId': self.file_id.replace('/', '$'),
#             'trunkId': trunk_id,
#             'sessionId': session_id
#         }
#         r = requests.get(self.host + ':' + str(self.port) + '/api/get-log', params=payload)
#         infos = json.loads(r.text)[0]

 #         features = {
#             'tag': tag,
#             'infer': self.inference_list[index]
#             'method': infos[1],
#             'host': infos[4],
#             'agent': infos[5],
#             'client_ip': infos[6]
#         }

 #         self.info_cache[index] = (tensor, features)
#         return tensor, features


class CSVMyDataset(cnn_dataset.MyDataset):
    def __init__(self, file_id):
        super(CSVMyDataset, self).__init__(file_id)
        self.metadata_list = file.read_session_metadata_file(util.get_session_metadata_path(file_id))
        self.guess_list = file.read_session_guess_file(util.get_session_guess_path(file_id))
        self.session_count = len(self.metadata_list)
        import pdb; pdb.set_trace()

     def __getitem__(self, index):
        tensor, tag = super(CSVMyDataset, self).__getitem__(index)
        trunk_id, session_id = self.file_map[index]

         metadata = self.metadata_list[index]
        guess = self.guess_list[index]
        features = {
            'tag': tag,
            # 'infer': self.inference_list[index] if self.inference_list else "-1",
            'infer': guess,
            'client_ip': metadata[0],
            'host': metadata[1],
            'user_agent': metadata[2]
        }
        return tensor, features