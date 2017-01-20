#!/usr/bin/env python3
import logging
import subprocess
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
from multiprocessing import Pool, cpu_count
from itertools import product
from functools import partial

env_tpl = '''
TRAVIS_REPO_SLUG={}
TRAVIS_PULL_REQUEST="false"
TRAVIS_COMMIT={}
'''

def _run_config(tag, clone_dir, commit):
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        subprocess.check_call(['git', 'clone', clone_dir, 'code'])
        os.chdir('code')
        subprocess.check_call(['git', 'checkout', commit])
        subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        image = 'dunecommunity/dune-gdt-testing:{}_master'.format(tag)
        subprocess.check_call(['docker', 'pull', image])
        with NamedTemporaryFile('wt') as envfile:
            envfile.write(env_tpl.format(slug, commit))
            cmd = ['docker', 'run', '--env-file', envfile.name, '-v', '{}:/root/src'.format(os.getcwd()),
                   image, './dune-gdt/.travis.script.bash']
            try:
                _ = subprocess.check_call(cmd)
            except subprocess.CalledProcessError as err:
                logging.error('Failed config: {}'.format(tag))
                logging.error(err)

docker_tags = ['gcc-5', 'clang-3.9']
slug = 'dune-community/dune-gdt-super'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

with TemporaryDirectory() as clone_tmp:
    clone_dir = os.path.join(clone_tmp, 'dune-gdt')
    subprocess.check_call(['git', 'clone', 'https://github.com/{}.git'.format(slug), clone_dir])
    run_configs = partial(_run_config, clone_dir=clone_dir, commit=commit)
    cpus = int(cpu_count()/2)
    with Pool(processes=cpus) as pool:
        _ = pool.map(run_configs, docker_tags)
