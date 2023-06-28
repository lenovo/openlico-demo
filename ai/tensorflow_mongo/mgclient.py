# Copyright 2015-2023 Lenovo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pymongo
import time

NAME_MAP = {'cnews.train': 'cnews.train.txt',
            'cnews.val': 'cnews.val.txt',
            'cnews.test': 'cnews.test.txt'}


class MongodbClient(object):
    def __init__(self, host='10.240.212.147', port=27017):

        self.client = pymongo.MongoClient(host, port)

    def get_tables(self):
        db = self.client['test2']
        tables = db.collection_names()
        return tables

    def get_value(self, db_name, table_name): 
        result = {}
        t1 = time.time()
        print 'start time:{}'.format(int(t1))
        for value in self.client[db_name][table_name].find():
            if not value:
                break
            filename = NAME_MAP[table_name]
            old_value = result.get(filename, '')
            if not old_value:
                result[filename] = [value['content']]
            else:
                result[filename].append(value['content'])
        result[NAME_MAP[table_name]] = ''.join(result[NAME_MAP[table_name]])
        print(result.keys())
        t2 = time.time()
        print 'end time:{}'.format(int(t2))
        print "duration: {0}".format(int(t2-t1))
        return result
if __name__ == '__main__':
    t1 = MongodbClient()
    t2=t1.get_value('test2','cnews.train')
    print len(t2)
