#!/bin/bash

# usage
if [ "$#" -lt "1" ]; then
  echo "usage: docker_push.sh [[registry]] [account]"
  exit 255
fi

# params
registry=""
account=${2:-$1}
if [ "$#" == "2" ]; then
    registry="${1}/"
fi

# do
branch=`git rev-parse --abbrev-ref HEAD`
if [ "${branch}" == "HEAD" ]; then branch=`git describe --tags`; fi

project=`basename \`pwd\``

echo -e "\033[1;32m[INFO]\033[m to tag ${registry}${account}/${project}:${branch}"
docker tag ${project}:${branch} ${registry}${account}/${project}:${branch}
echo -e "\033[1;32m[INFO]\033[m to push ${registry}${account}/${project}:${branch}"
docker push ${registry}${account}/${project}:${branch}

echo -e "\033[1;32m[INFO]\033[m to tag ${registry}${account}/${project}:latest"
docker tag ${project}:${branch} ${registry}${account}/${project}:latest
echo -e "\033[1;32m[INFO]\033[m to push ${registry}${account}/${project}:latest"
docker push ${registry}${account}/${project}:latest
