#!/bin/bash

ps_num=2
worker_num=8
B=3


if [ -z "$1" ]; then
mkdir data
for i in `eval echo {0..$ps_num}`
do
  python doom_pathnet.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=ps --task_index=$i --B=$B &
done

for i in `eval echo {0..$worker_num}`
do
  unbuffer python doom_pathnet.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=worker --task_index=$i --B=$B > >(tee -a data/worker"$i".out) 2> >(tee -a data/worker"$i".err >&2) &
done

else
for i in `eval echo {0..$ps_num}`
do
  python doom_pathnet.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=ps --task_index=$i --B=$B --log_dir="$1" &
done

for i in `eval echo {0..$worker_num}`
do
  unbuffer python doom_pathnet.py --ps_hosts_num=$ps_num --worker_hosts_num=$worker_num --job_name=worker --task_index=$i --B=$B --log_dir="$1" > >(tee -a data/worker"$i".out) 2> >(tee -a data/worker"$i".err >&2) &
done
fi

