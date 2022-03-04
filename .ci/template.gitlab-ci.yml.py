#!/usr/bin/env python3

import os
import jinja2
from pathlib import Path

# from dotenv import dotenv_values
from itertools import product

tpl = r"""# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/template.gitlab-ci.yml.py instead       #

stages:
  - sanity
  - base
  # - modules
  - wheels
  - test
  - publish

variables:
  GIT_SUBMODULE_STRATEGY: none
  ML_TAG: 964e0675bf86ec15c06de2e10003b02d9688ec07
  WHEEL_DIR: ${CI_PROJECT_DIR}/wheelhouse
  DUNE_BUILD_DIR: ${CI_PROJECT_DIR}/build
  DUNE_SRC_DIR: ${CI_PROJECT_DIR}

.docker-in-docker:
    retry:
        max: 2
        when:
            - runner_system_failure
            - api_failure
            - unknown_failure
    image: docker:stable
    variables:
        DOCKER_HOST: tcp://docker:2375/
        DOCKER_DRIVER: overlay2
    before_script:
        - apk --update add py3-pip openssh-client rsync git file bash python3 py3-cffi py3-cryptography
        - pip3 install -U docker jinja2 docopt twine
        - echo $DOCKER_PW | docker login --username="$DOCKER_USER" --password-stdin
        - git submodule update --init .ci/shared
        - 'export SHARED_PATH="${CI_PROJECT_DIR}/shared"'
        - mkdir -p ${SHARED_PATH}
    services:
        - docker:dind
    environment:
        name: unsafe
    tags:
      - dind
      - amm-only
    # stage: modules
    script: .ci/shared/docker/update_test_dockers.py ${MODULE_NAME}

sanity:
  stage: sanity
  image: harbor.uni-muenster.de/proxy-docker/library/alpine:3.15
  before_script:
    - apk add git python3
    - pip3 install jinja2
  script:
    - ./.ci/template.gitlab-ci.py && git diff --exit-code .ci/gitlab-ci.yml

base:
    extends: .docker-in-docker
    stage: base
    variables:
      MODULE_NAME: BASE

# gdt:
#     extends: .docker-in-docker
#     variables:
#       MODULE_NAME: dune-gdt

# super:
#   extends: .docker-in-docker
#   variables:
#     BASE: debian
#     IMAGE: dunecommunity/gdt-super_${BASE}:${CI_COMMIT_SHA}
#   script: |
#     docker build --build-arg BASE=${BASE} -t ${IMAGE} -f .ci/shared/docker/super_docker/Dockerfile .
#     docker push ${IMAGE}

.wheels_base:
  variables:
    GIT_SUBMODULE_STRATEGY: recursive
    LOCAL_UID: 0
    TWINE_PASSWORD: ${CI_JOB_TOKEN}
    TWINE_USERNAME: gitlab-ci-token
    GITLAB_PYPI: ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi/
  stage: wheels
  image: zivgitlab.wwu.io/ag-ohlberger/dune-community/docker/manylinux-2014_py${GDT_PYTHON_VERSION}:${ML_TAG}
  needs: []
  artifacts:
    paths:
      - ${WHEEL_DIR}/final/*.whl
      - ${DUNE_BUILD_DIR}

.test_base:
  image: pymor/testing_py${GDT_PYTHON_VERSION}:latest
  stage: test
  script:
    - pip install ${WHEEL_DIR}/final/dune*whl
    - python -c 'from dune.${md} import *'

{% for py in pythons %}

{% for md in wheel_steps %}
{{md}} {{py}}:
  extends: .wheels_base
  variables:
    GDT_PYTHON_VERSION: "{{py}}"
  cache:
    key: $CI_COMMIT_REF_SLUG_{{py}}
{% if not loop.first %}
  needs: [{{loop.previtem}} {{py}}]
{% endif %}
  script:
  - ./make_wheels.bash {{md}}  
{% if md != "all" %}
  - python3 -m twine check ${WHEEL_DIR}/final/*{{md}}*.whl
  - python3 -m twine upload --repository-url ${GITLAB_PYPI} ${WHEEL_DIR}/final/*{{md}}*.whl
{% endif %}
{% endfor %}

test wheels {{py}}:
  extends: .test_base
  needs: ["gdt {{py}}"]
  dependencies: ["gdt {{py}}"]

{% endfor %}

.publish:
  image: alpine:3.15
  dependencies:
{% for py in pythons %}  
  -  "gdt {{py}}"
{% endfor %}
  needs: 
{% for py in pythons %}  
  -  "gdt {{py}}"
  -  "test wheels {{py}}"
{% endfor %}
  stage: publish
  before_script:
      - apk --update add py3-pip git file bash python3 py3-cffi py3-cryptography
      - pip3 install -U twine
  variables:
    TWINE_PASSWORD: ${TWINE_PASSWORD}
    TWINE_USERNAME: ${TWINE_USERNAME}
  script: 
    - cd ${MOD_DIR}
    # upload only if a tag points to checked out commit
    - (git describe --exact-match --tags HEAD && python3 -m twine upload ${WHEEL_DIR}/final/${MOD_WHEEL_PREFIX}*whl) || echo "not uploading untagged wheels"
  artifacts:
    paths:
      - ${WHEEL_DIR}/final/${MOD_WHEEL_PREFIX}*whl
    
publish dune-xt:
  extends: .publish
  variables:
    MOD_WHEEL_PREFIX: dune_xt
    MOD_DIR: dune-xt

publish dune-gdt:
  extends: .publish
  variables:
    MOD_WHEEL_PREFIX: dune_gdt
    MOD_DIR: dune-gdt


# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/template.gitlab-ci.yml.py instead       #

"""


tpl = jinja2.Template(tpl)
pythons = ["3.7", "3.8", "3.9"]
wheel_steps = ["all", "xt", "gdt"]
# env_path = Path(os.path.dirname(__file__)) / '..' / '..' / '.env'
# env = dotenv_values(env_path)
# ci_image_tag = env['CI_IMAGE_TAG']
# pypi_mirror_tag = env['PYPI_MIRROR_TAG']

with open(os.path.join(os.path.dirname(__file__), "gitlab-ci.yml"), "wt") as yml:
    yml.write(tpl.render(**locals()))
