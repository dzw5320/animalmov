#!/bin/bash

for args in `seq 1 100`;
do
qsub 020_ex5version.PBS -v "args=$args"
done
