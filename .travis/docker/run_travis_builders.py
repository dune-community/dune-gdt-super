#!/usr/bin/env python3
import logging
import subprocess
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
from multiprocessing import Pool, cpu_count
from itertools import product
from functools import partial
import sys
import update_image

env_tpl = '''
TRAVIS_REPO_SLUG={}
TRAVIS_PULL_REQUEST="false"
TRAVIS_COMMIT={}
MY_MODULE=dune-gdt
'''


def _cmd(cmd, logger):
    logger.info(' '.join(cmd))
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    logger.debug(out)


def _run_config(tag, clone_dir, commit, level=logging.DEBUG):
    logger = logging.getLogger(tag)
    myFormatter = logging.Formatter('{}: %(asctime)s - %(message)s'.format(tag))
    handler = logging.StreamHandler()
    handler.setFormatter(myFormatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    update_image.update(commit, tag)

    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        _cmd(['git', 'clone', clone_dir, 'code'], logger)
        os.chdir('code')
        _cmd(['git', 'checkout', commit], logger)
        _cmd(['git', 'submodule', 'update', '--init', '--recursive'], logger)
        image = 'dunecommunity/dune-gdt-testing_{}:master'.format(tag)
        _cmd(['docker', 'pull', image], logger)
        with NamedTemporaryFile('wt') as envfile:
            envfile.write(env_tpl.format(slug, commit))
            cmd = ['docker', 'run', '--env-file', envfile.name, '-v', '{}:/root/src/dune-gdt'.format(os.getcwd()),
                   image, '/root/src/dune-gdt/.travis.script.bash']
            try:
                _ = _cmd(cmd, logger)
            except subprocess.CalledProcessError as err:
                logging.error('Failed config: {}'.format(tag))
                logging.error(err)
                logging.error(err.output)

docker_tags = ['gcc', 'clang']
slug = 'dune-community/dune-gdt'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
level = logging.DEBUG if '-v' in sys.argv else logging.INFO
logging.basicConfig()
with TemporaryDirectory() as clone_tmp:
    clone_dir = os.path.join(clone_tmp, 'dune-gdt')
    _cmd(['git', 'clone', 'https://github.com/{}.git'.format(slug), clone_dir], logging)
    cpus = int(cpu_count()/2)
    run_configs = partial(_run_config, clone_dir=clone_dir, commit=commit, level=level)
    with Pool(processes=cpus) as pool:
        _ = pool.map(run_configs, docker_tags)
