#!/bin/bash


make


for (( i=1; i <= 100; ++i ))
do
   bin/./example_ozenn
   echo "i= ", $i
done
