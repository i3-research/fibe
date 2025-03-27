#!/bin/bash

branch=`git rev-parse --abbrev-ref HEAD`
if [ "${branch}" == "HEAD" ]; then branch=`git describe --tags`; fi

project=`basename \`pwd\``

echo -e "\033[1;32m[INFO]\033[m to build ${project}:${branch}"
docker build -t ${project}:${branch} .
