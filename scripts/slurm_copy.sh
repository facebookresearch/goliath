#!/bin/bash

###-----------------------------------------------------------------
### Configuration variables
declare -a A=("a" "b")
declare -a B=("c" "d")

for (( i=0; i<"${#A[@]}"; i++ ));
do
    echo "$i ${A[i]} ${B[i]}"
done
