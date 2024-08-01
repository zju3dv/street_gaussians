#!/bin/sh

protoc -I=. --python_out=. simple_waymo_open_dataset_reader/label.proto
protoc -I=. --python_out=. simple_waymo_open_dataset_reader/dataset.proto

