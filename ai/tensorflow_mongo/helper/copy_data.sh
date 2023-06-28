#!/bin/bash

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

# copy MAXCOUNT files from each directory

MAXCOUNT=6500

for category in $( ls THUCNews); do
  echo item: $category

  dir=THUCNews/$category
  newdir=data/thucnews/$category
  if [ -d $newdir ]; then
    rm -rf $newdir
    mkdir $newdir
  fi

  COUNTER=1
  for i in $(ls $dir); do
    cp $dir/$i $newdir
    if [ $COUNTER -ge $MAXCOUNT ]
    then
      echo finished
      break
    fi
    let COUNTER=COUNTER+1
  done

done
