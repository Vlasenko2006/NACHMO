#!/bin/bash


make


for (( i=1; i <= 10; ++i ))
do
   bin/./example_ozenn
   echo $i
done
