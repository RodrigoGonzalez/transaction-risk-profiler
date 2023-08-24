# =============================================================================
# MAKEFILE FOR Fraud Detection
# =============================================================================
#
# Partially inspired by https://github.com/johschmidt42/python-project-johannes
#
# To do stuff with make, you type `make` in a directory that has a file called
# "Makefile". You can also type `make -f <makefile>` to use a different filename.
#
# A Makefile is a collection of rules. Each rule is a recipe to do a specific
# thing, sort of like a grunt task or an npm package.json script.
#
# A rule looks like this:
#
# <target>: <prerequisites...>
# 	<commands>
#
# The "target" is required. The prerequisites are optional, and the commands
# are also optional, but you have to have one or the other.
#
# Type `make` to show the available targets and a description of each.
#

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

PROJECTNAME := $(shell basename "$(PWD)")
PYTHON_INTERPRETER := python3.11
COMMIT_ID=$(shell git rev-parse HEAD)
DEFAULT_BRANCH := master

.SILENT: ;               # no need for @

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

##@ Environment

setup: poetry-install pre-commit-install  ## Setup Virtual Environment

    poetry-install:  ## Install dependencies using Poetry
		poetry env use $(PYTHON_INTERPRETER)
		poetry install
		poetry self add poetry-plugin-up

    pre-commit-install:  ## Install pre-commit hooks
		poetry run pre-commit install

.PHONY: setup poetry-install pre-commit-install

update-deps: pip-upgrade poetry-update pre-commit-autoupdate  ## Update dependencies

    pip-upgrade:  ## Upgrade pip
		poetry run pip install --upgrade pip

    poetry-update:  ## Update Poetry dependencies
		poetry self update
		poetry update
		poetry lock

    pre-commit-autoupdate:  ## Update pre-commit hooks
		poetry run pre-commit autoupdate -c .pre-commit-config.yaml

upgrade-deps: update-deps ## Upgrade dependencies to the latest versions
	# https://github.com/MousaZeidBaker/poetry-plugin-up
	poetry up


.PHONY: update-deps upgrade-deps local pip-upgrade poetry-update pre-commit-autoupdate

local: setup update-deps upgrade-deps  ## Locally install the package
	discovergpt --help

activate:  ## Activate the virtual environment
	poetry shell

.PHONY: update-deps upgrade-deps local pip-upgrade poetry-update pre-commit-autoupdate

# =============================================================================
# DEVELOPMENT
# =============================================================================

##@ Development

add-all:  ## Add all files to git
	# not added to `pre-commit-tool` in order to prevent unwanted behavior when running in workflows
	git add -A

pre-commit: add-all  ## Manually run all pre-commit hooks
	export PYTHONPATH=$PYTHONPATH:/Users/rodrigogonzalez/workspace/DiscoverGPT
	poetry run pre-commit run -c .pre-commit-config.yaml

pre-commit-tool:  ## Manually run a single pre-commit hook (e.g. `make pre-commit-tool TOOL=black`)
	poetry run pre-commit run --hook-stage manual $(TOOL) -c .pre-commit-config.yaml

# https://commitizen-tools.github.io/commitizen/bump/
#commit: pre-commit tests  ## Commit changes
commit: pre-commit  ## Commit changes
	./scripts/commit.sh

bump:  ## Bump version and update changelog
	poetry run cz bump --changelog --check-consistency --annotated-tag --retry
	git push -u origin HEAD --follow-tags

.PHONY: pre-commit pre-commit-tool commit bump


# =============================================================================
# TESTING
# =============================================================================

##@ Testing

tox: ## run tox tests
	poetry run tox
	make clean

tests: unit-tests  ## run all tests

unit-tests: ## run unit-tests with pytest
	poetry run pytest  -vvvvsra --doctest-modules

unit-tests-cov: ## run unit-tests with pytest and show coverage (terminal + html)
	poetry run pytest  -vvvvsra --doctest-modules --cov=src --cov-report term-missing --cov-report=html

unit-tests-cov-fail: ## run unit tests w/ pytest and coverage (terminal + html) & create files for CI
	poetry run pytest  -vvvvsra --doctest-modules --cov=src --cov-report term-missing \
	--cov-report=xml --cov-fail-under=80 --junitxml=pytest.xml | tee pytest-coverage.txt

clean-cov: ## remove output files from pytest & coverage
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov
	rm -rf pytest.xml
	rm -rf pytest-coverage.txt
	rm -rf dist

.PHONY: tox tests unit-tests unit-tests-cov unit-tests-cov-fail clean-cov

# =============================================================================
# DOCUMENTATION
# =============================================================================

##@ Documentation

docs-serve: ## serve documentation locally
	mkdocs serve

docs-build: ## build documentation locally
	mkdocs build

docs-deploy: ## build & deploy documentation to "gh-pages" branch
	mkdocs gh-deploy -m "docs: update documentation" -v --force

clean-docs: ## remove output files from mkdocs
	rm -rf site

.PHONY: docs-serve docs-build docs-deploy clean-docs

# =============================================================================
# BUILD & RELEASE
# =============================================================================

##@ Build & Release

clean: clean-docs  clean-cov  ## Clean package
	find . -type d -name '__pycache__' | xargs rm -rf
	find . -type d -name '.temp' | xargs rm -rf
	find . -type f -name '.coverage' | xargs rm -rf
	rm -rf build dist

build:  pre-commit clean  ## Build the project
	poetry build

deploy:  ## Deploy to PyPI
	poetry publish --build

.PHONY: build clean deploy

# -----------------------------------------------------------------------------
# GIT
# -----------------------------------------------------------------------------

##@ Git diff Shortcuts

diff-name:  ## Show diff name
	@ echo "Running git diff --name-only command:\n" && \
 	echo "Files changed:"
	@ git --no-pager diff --name-only $(shell git merge-base origin/$(DEFAULT_BRANCH) HEAD)

diff-stat:  ## Show diff stat
	@ echo "Running git diff --stat command:\n" && \
 	echo " Files Changed 	| Lines added "
	@ git --no-pager diff --stat $(shell git merge-base origin/$(DEFAULT_BRANCH) HEAD)

diff-num:  ## Show diff numstat
	@ echo "Running git diff --numstat command:\n" && \
 	echo "Lines added | Lines removed | File"
	@ git --no-pager diff --numstat $(shell git merge-base origin/$(DEFAULT_BRANCH) HEAD)

diff-summary:  ## Show diff numstat
	@ echo "Running git diff --summary command:\n"
	@ git --no-pager diff --summary $(shell git merge-base origin/$(DEFAULT_BRANCH) HEAD)

list-branches:  ## List branches
	@ echo "Running git branch --sort=-committerdate:\n"
	@ git --no-pager branch --sort=-committerdate


.PHONY: diff-name diff-stat diff-num list-branches

##@ Git Shortcuts

git-commit-num:
	@echo "+ $@"
	@echo $(shell git rev-list --all --count)

.PHONY: git-commit-num

current_branch := $(shell git symbolic-ref --short HEAD)

checkout-main:  ## Switch to main branch
	@echo "+ $@"
	if [ "$(current_branch)" != $(DEFAULT_BRANCH) ]; then \
		git checkout $(DEFAULT_BRANCH); \
	fi
	git pull --all
	git fetch --tags

.PHONY: checkout-main

commit_count := $(shell git rev-list --all --count)

check-branch-name:
ifeq ($(BRANCH),)
	$(error BRANCH variable is not set. Please provide a branch name with BRANCH=mybranch)
endif

new-branch: check-branch-name  ## Create a new branch
	git checkout -b $(BRANCH)_$(commit_count)

new-feat-branch: check-branch-name  ## Create a new feature branch
	git checkout -b feat/$(BRANCH)_$(commit_count)

new-version-branch:  ## Create a new version branch
	NEW_VERSION=$(shell poetry run cz bump --dry-run | grep 'bump: version' | awk -F ' ' '{print $$NF}'); \
	git checkout -b v$$NEW_VERSION

.PHONY: check-branch-name new-branch new-feat-branch new-version-branch

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

##@ Docker Shortcuts

docker-version:  ## Show docker version
	@echo "+ $@"
	@docker --version

up:  ## Start current docker-compose
	@echo "+ $@"
	@docker-compose up -d --remove-orphans

up-build:  ## Start current docker-compose
	@echo "+ $@"
	@docker-compose up -d --remove-orphans --build

up-build-no-cache:
	@echo "+ $@"
	@docker-compose build --no-cache --force-rm backend celeryworker
	@docker-composeup up -d --remove-orphans

up-dev:  ## Start development docker-compose
	@echo "+ $@"
	@docker-compose -f docker-compose-dev.yml -f docker-compose-dev.override.yml up -d --remove-orphans --build

up-dev-no-cache:  ## Start development docker-compose
	@echo "+ $@"
	@docker-compose -f docker-compose-dev.yml -f docker-compose-dev.override.yml build --no-cache --force-rm backend celeryworker
	@docker-compose -f docker-compose-dev.yml -f docker-compose-dev.override.yml up -d --remove-orphans

down:  ## Stop docker-compose
	@echo "+ $@"
	@docker-compose down

down-dev:  ## Stop development docker-compose
	@echo "+ $@"
	@docker-compose -f docker-compose-dev.yml -f docker-compose-dev.override.yml down

enter-backend:  ## Enter backend container
	@echo "+ $@"
	# @docker exec -it $(shell docker ps -aqf "name=$(PROJECTNAME)_$(SERVICE)_1") /bin/bash
	@docker-compose exec backend bash

quick-check:  ## Run quick check
	@echo "+ $@"
	@docker-compose exec backend ./scripts/test.sh -vvvvsra --doctest-modules

#wait-for-services:
#	@echo "+ $@"
#	@./scripts/wait-for-services.sh
#
#```
##!/bin/bash
#
#while ! mysqladmin ping -h database --silent; do
#    sleep 1
#    echo "Waiting for database to be ready..."
#done
#```
#
#long-check: down up wait-for-services ## Run long check (rebuilds containers)
#	@echo "+ $@"
#	@docker-compose exec backend ./scripts/test.sh -vvvvsra --doctest-modules

.PHONY: docker-version up build-no-cache up-dev up-dev-no-cache down enter-backend quick-check

# =============================================================================
# ADDITIONAL
# =============================================================================

# =============================================================================
# SELF DOCUMENTATION
# =============================================================================

##@ Help

.DEFAULT_GOAL := help
.PHONY: help

help:  ## Display this help
	echo
	echo " The following commands can be run for "$(PROJECTNAME)":"
	echo
	awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ \
	{ printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' \
	$(MAKEFILE_LIST)
