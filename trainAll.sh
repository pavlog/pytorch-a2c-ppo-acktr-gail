#!/bin/bash
while :
do
  python3 main.py --action-type=0  
  sleep 10
  python3 main.py --action-type=1  
  sleep 10
done
