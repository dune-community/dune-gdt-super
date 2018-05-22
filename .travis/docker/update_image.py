#!/usr/bin/env python3

import os
import subprocess
import sys
import six
import re
try:
    import docker
except ImportError:
    print('missing module: pip install docker')
    sys.exit(1)
from docker.utils.json_stream import json_stream

cc_mapping = {'gcc': 'g++', 'clang': 'clang++'}
thisdir = os.path.dirname(os.path.abspath(__file__))

def _build(client, **kwargs):
    resp = client.api.build(**kwargs)
    if isinstance(resp, six.string_types):
        return client.images.get(resp)
    last_event = None
    image_id = None
    output = []
    for chunk in json_stream(resp):
        if 'error' in chunk:
            msg = chunk['error'] + '\n' + ''.join(output)
            raise docker.errors.BuildError(msg, chunk)
        if 'stream' in chunk:
            output.append(chunk['stream'])
            match = re.search(
                r'(^Successfully built |sha256:)([0-9a-f]+)$',
                chunk['stream']
            )
            if match:
                image_id = match.group(2)
        last_event = chunk
    if image_id:
        return client.images.get(image_id)
    raise docker.errors.BuildError(last_event or 'Unknown', '')

def update(commit, cc):
    gdt_super_dir = os.path.join(thisdir, '..', '..',)
    dockerfile = os.path.join(thisdir, 'dune-gdt-testing', 'Dockerfile')
    client = docker.from_env(version='auto')

    os.chdir(gdt_super_dir)

    cxx = cc_mapping[cc]
    commit = commit.replace('/', '_')
    repo = 'dunecommunity/dune-gdt-testing_{}'.format(cc)
    buildargs = {'cc': cc, 'cxx': cxx, 'commit': commit }
    tag = '{}:{}'.format(repo, commit)
    img = _build(client, rm=True, fileobj=open(dockerfile, 'rb'),
                        tag=tag, buildargs=buildargs, nocache=False)
    #img.tag(repo, refname)
    client.images.push(repo)

    try:
        client.images.remove(img.id, force=True)
    except docker.errors.APIError as err:
        logging.error('Could not delete {} - {} : {}'.format(img.name, img.id, str(err)))


if __name__ == '__main__':
    if len(sys.argv) > 2:
        ccs = [sys.argv[1]]
        commits = [sys.argv[2]]
    else:
        ccs = list(cc_mapping.keys())
        commits = ['master']

    subprocess.check_call(['docker', 'pull', 'dunecommunity/testing-base:latest'])
    for b in commits:
        for c in ccs:
            update(b, c)
    subprocess.check_call(['docker', 'images'])
