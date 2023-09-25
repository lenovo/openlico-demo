# Copyright 2015-present Lenovo
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

import happybase

NAME_MAP = {
    'train': 'cnews.train.txt',
    'val': 'cnews.val.txt',
    'test': 'cnews.test.txt'
}


class HbaseClient(object):
    def __init__(self, host='10.240.208.136', port=9090):
        self.transport = happybase.Connection(
            host,
            port,
            autoconnect=False,
            timeout=10*60*1000
        )

    def get_tables(self):
        self.transport.open()
        tables = self.transport.tables()
        self.transport.close()
        return tables

    def get_value(self, table_name, row, column_cluster='yanxiao'):
        scope = 500
        result = {}
        self.transport.open()
        table = self.transport.table(table_name)

        for column, filename in NAME_MAP.items():
            for s in range(scope):
                query_column = '{0}:{1}-{2}'.format(column_cluster, column, s)
                ret_dict = table.row(row, [query_column])
                value = ret_dict.get(query_column.encode())
                if not value:
                    break
                if not isinstance(value, str):
                    value = value.decode()
                old_value = result.get(filename, '')
                if not old_value:
                    result[filename] = value
                else:
                    result[filename] += value

        self.transport.close()
        return result
