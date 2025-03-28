#!/bin/bash

# virtualenv_dir
virtualenv_dir="${BASH_ARGV[0]:-.venv}"

# cc_dir
cc_dir="${BASH_ARGV[1]:-.cc}"

the_basename=`basename \`pwd\``

echo -e "\033[1;32m[INFO]\033[m virtualenv_dir: ${virtualenv_dir} cc dir: ${cc_dir} the_basename: ${the_basename}"

# virtualenv
if [ ! -d ${virtualenv_dir} ]
then
  echo -e "\033[1;32m[INFO]\033[m no ${virtualenv_dir}. will create one"
  virtualenv -p `which python3` --prompt="${the_basename}" "${virtualenv_dir}"
fi

echo -e "\033[1;32m[INFO]\033[m to activate: ${virtualenv_dir}"
source ${virtualenv_dir}/bin/activate

the_python_path=`which python`
echo -e "\033[1;32m[INFO]\033[m python: ${the_python_path}"

echo -e "\033[1;32m[INFO]\033[m current_dir: `pwd`"

# gitignore
if [ ! -f .gitignore ]
then
    echo "/${virtualenv_dir}" >> .gitignore
fi

# requirements-dev.txt
echo -e "\033[1;32m[INFO]\033[m requirements-dev:"
if [ ! -f requirements-dev.txt ]
then
    cp ${cc_dir}/requirements-dev.txt requirements-dev.txt
fi
pip install -r requirements-dev.txt

# remove ${cc_dir}/.git
if [ -e ${cc_dir}/.git ]; then
    rm -rf ${cc_dir}/.git*
fi

# .cc/scripts
if [ ! -e scripts ]; then
    mkdir -p scripts
    cp ${cc_dir}/scripts/*.sh ./scripts
fi

if [ -e .git ]; then
    echo -e "\033[1;32m[INFO]\033[m .git exists. assuming no need to init project."
    echo -e "\033[1;32m[INFO]\033[m remember to: \033[1;32m. ${virtualenv_dir}/bin/activate\033[m"
    exit 0
fi

# project_dev
echo -e "\033[1;32m[INFO]\033[m to init project"
if [ ! -e ${the_basename} ]; then
    ./scripts/project_dev.sh
fi

# git init
if [ ! -e .git ]; then
    echo -e "\033[1;32m[INFO]\033[m to git init"
    git init; git add .; git commit -m "init dev"
fi

# done
echo -e "\033[1;32m[INFO]\033[m done"
echo -e "\033[1;32m[INFO]\033[m remember to: \033[1;32msource ${virtualenv_dir}/bin/activate\033[m"
