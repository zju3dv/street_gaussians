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

import os
from simple_waymo_open_dataset_reader import WaymoDataFileReader

path = os.path.expanduser("~/data/waymo/training/")

total_num_frames = 0

for entry in os.listdir(path):
    filename = os.path.join(path,entry)
    datafile = WaymoDataFileReader(filename)
    table = datafile.get_record_table()

    num_frames = len(table)
    print(os.path.splitext(entry)[0],num_frames)

    total_num_frames += num_frames

print("Total", total_num_frames)

