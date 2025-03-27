#!/bin/bash

echo -e "\x1b[1;32m[INFO]\x1b[m to flake8 pyutil_cfg tests"
flake8 .
echo -e "\x1b[1;32m[INFO]\x1b[m flake8 done"
