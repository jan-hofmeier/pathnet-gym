#!/bin/bash

ps_num=2
worker_num=4
B=2

for i in `eval echo {0..$ps_num}`
do
  python doom_pathnet.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=ps --task_index=$i --B=$B &
done

mkdir data
for i in `eval echo {0..$worker_num}`
do
  python doom_pathnet.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=worker --task_index=$i --B=$B > >(tee -a data/worker"$i".out) 2> >(tee -a data/worker"$i".err >&2) &
done
