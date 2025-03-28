# cc-python-template

cookiecutter python template (pkgs, modules and tests)

    git clone https://github.com/chhsiao1981/cc-python-template.git .cc && ./.cc/scripts/init_dev.sh && source .venv/bin/activate

* create module: ./scripts/dev_module.sh

## Introduction

This template intends to facilitate the efficient development of Python projects with [cookiecutter](https://github.com/cookiecutter/cookiecutter).
All are welcome to improve this template.

As specified in [gen.py](https://github.com/chhsiao1981/cc-python-template/blob/main/gen.py),
currently the following variables are defined (using `a.b.c.d` in the project directory `awesome-project` as an example):

* `pkg`: (ex: `awesome_project.a.b.c`)
* `module`: (ex: `d`)
* `pkg_name`: (ex: `awesome_project.a.b.c`)
* `project_name`: (ex: `awesome_project`)
* `project_name_with_dash`: (ex: `awesome-project`)
* `include_pkg`: (ex: `awesome_project.a.b.c`)
* `package_dir`: (ex: `awesome_project/a/b/c`)
* `include_package_dir`: (ex: `awesome_project/a/b/c`)
* `test_package_dir`: (ex: `test_a/test_b/test_c`)

UPPERCASE:
* `PKG`: UPPERCASE of `pkg` (ex: `AWESOME_PROJECT.A.B.C`)
* `MODULE`: UPPERCASE of `module` (ex: `D`)
* `PROJECT` UpperCamelCase of `project_name` (ex: `AwesomeProject`)
* `PKG_NAME`: UPPERCASE of `pkg_name` (ex: `AWESOME_PROJECT.A.B.C`)
* `PROJECT_NAME` UpperCamelCase of `project_name` (ex: `AwesomeProject`)
* `INCLUDE_PKG`: UPPERCASE of `include_pkg` (ex: `AWESOME_PROJECT.A.B.C`)
* `PACKAGE_DIR`: UPPERCASE of `package_dir` (ex: `AWESOME_PROJECT/A/B/C`)
* `include_package_dir`: UPPERCASE of `include_package_dir` (ex: `AWESOME_PROJECT/A/B/C`)
* `test_package_dir`: UPPERCASE of `test_package_dir` (ex: `TEST_A/TEST_B/TEST_C`)

UpperCamelCase:
* `Pkg`: UpperCamelCase of `pkg` (ex: `AwesomeProject.A.B.C`)
* `Module`: UpperCamelCase of `module` (ex: `D`)
* `Project` UpperCamelCase of `project_name` (ex: `AwesomeProject`)
* `PkgName`: UpperCamelCase of `pkg_name` (ex: `AwesomeProject.A.B.C`)
* `ProjectName` UpperCamelCase of `project_name` (ex: `AwesomeProject`)
* `IncludePkg`: UpperCamelCase of `include_pkg` (ex: `AwesomeProject.A.B.C`)
* `PackageDir`: UpperCamelCase of `package_dir` (ex: `AwesomeProject/A/B/C`)
* `IncludePackageDir`: UpperCamelCase of `include_package_dir` (ex: `AwesomeProject/A/B/C`)
* `TestPackageDir`: UpperCamelCase of `test_package_dir` (ex: `TestA/TestB/TestC`)

camelCase:
* `pkgCamel`: camelCase of `pkg` (ex: `awesomeProject.A.B.C`)
* `moduleCamel`: camelCase of `module` (ex: `D`)
* `projectCamel` camelCase of `project_name` (ex: `awesomeProject`)
* `pkgName`: camelCase of `pkg_name` (ex: `awesomeProject.A.B.C`)
* `projectName` camelCase of `project_name` (ex: `awesomeProject`)
* `includePkg`: camelCase of `include_pkg` (ex: `awesomeProject.A.B.C`)
* `packageDir`: camelCase of `package_dir` (ex: `awesomeProject/A/B/C`)
* `includePackageDir`: camelCase of `include_package_dir` (ex: `awesomeProject/A/B/C`)
* `testPackageDir`: camelCase of `test_package_dir` (ex: `testA/testB/testC`)

