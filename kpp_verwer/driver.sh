#!/bin/bash

amount_of_data=11

path="/scratch/g/g260141/results_verwer/result"

touch tmp.nml 

for(( i=0; i<=$amount_of_data; i++ ))
do


    path1=${path}$i


  echo $path1

  if [ -d "$path1" ]; then
    # Take action if $DIR exists. #
    echo "Directory ${path1} exists"
  else 
    echo "Directory ${path1} does not exist!"
    mkdir ${path1}
  fi



   echo $i > tmp.nml
   echo "LOOP $i "
   ./verwer.exe
done

