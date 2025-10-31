#!/bin/bash
num1=501
num2=600

if [ $num1 -gt $num2 ]; then
  echo "The first number must be less than or equal to the second number."
  exit 1
fi

for ((i = num1; i <= num2; i++))
do

cd ./expand_dataset/raw/$i
yhbatch job.sh
cd ../../../
done
    