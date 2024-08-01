# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
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
# ==============================================================================

import struct
from . import dataset_pb2

class WaymoDataFileReader:
    def __init__(self, filename):
        self.file = open(filename, "rb")

    def get_record_table(self):
        """ Generate and return a table of the offset of all frame records in the file.

            This is particularly useful to determine the number of frames in the file
            and access random frames rather than read the file sequentially.
        """

        self.file.seek(0,0)

        table = []

        while self.file:
            offset = self.file.tell()

            try:
                self.read_record(header_only=True)
                table.append(offset)
            except StopIteration:
                break

        self.file.seek(0,0)

        return table
    
    def seek(self, offset):
        """ Seek to a specific frame record by offset.

        The offset of each frame in the file can be obtained with the function reader.get_record_table()
        """

        self.file.seek(offset,0)

    def read_record(self, header_only = False):
        """ Read the current frame record in the file.

        If repeatedly called, it will return sequential records until the end of file. When the end is reached, it will raise a StopIteration exception.
        To reset to the first frame, call reader.seek(0)
        """
        
        # TODO: Check CRCs.

        header = self.file.read(12)

        if header == b'':
            raise StopIteration()

        length, lengthcrc = struct.unpack("QI", header)


        if header_only:
            # Skip length+4 bytes ahead
            self.file.seek(length+4,1)
            return None
        else:
            data = self.file.read(length)
            datacrc = struct.unpack("I",self.file.read(4))

            frame = dataset_pb2.Frame()
            frame.ParseFromString(data)
            return frame

    def __iter__(self):
        """ Simple iterator through the file. Note that the iterator will iterate from the current position, does not support concurrent iterators and will not reset back to the beginning when the end is reached. To reset to the first frame, call reader.seek(0)
        """
        return self

    def __next__(self):
        return self.read_record()


