import os
import platform
import random
import shlex
import string
import subprocess
import sys
import time

import blindspin
import fire
import yaml
from azure.common.credentials import ServicePrincipalCredentials
from azure.mgmt import batchai
from azure.mgmt.batchai import models


class Job:
    def __init__(self, config='./batchai.yaml'):
        with open(config) as fi:
            cfg = yaml.load(fi)

        self.cfg = cfg
        self.azure_cfg = cfg['azure']
        self.container_cfg = cfg['container']
        self.git_cfg = cfg['git']
        self.client = self._make_batchai_client()
        self._cluster = self.client.clusters.get(
            self.azure_cfg['resource_group'],
            self.azure_cfg['workspace_name'],
            self.azure_cfg['cluster_name'])

    def _make_batchai_client(self):
        return batchai.BatchAIManagementClient(
            subscription_id=self.azure_cfg['subscription_id'],
            credentials=ServicePrincipalCredentials(
                client_id=self.azure_cfg['aad_client_id'],
                secret=self.azure_cfg['aad_secret_key'],
                token_uri=(
                    'https://login.microsoftonline.com/{0}/oauth2/token'
                    .format(self.azure_cfg['aad_tenant_id']))))

    @property
    def _container_settings(self):
        return models.ContainerSettings(
            image_source_registry=models.ImageSourceRegistry(
                image=self.container_cfg['image'],
                server_url=self.container_cfg['registry_url'],
                credentials=models.PrivateRegistryCredentials(
                    username=self.container_cfg['username'],
                    password=self.container_cfg['password'],
                )
            )
        )

    def _prepare_job(self, command, job_node_count, branch, real_job_name):
        prefix = os.path.join('$AZ_BATCHAI_MOUNT_ROOT',
                              self.azure_cfg['azure_file_share'])

        output_directories = [models.OutputDirectory(
                id='OUT', path_prefix=prefix, path_suffix='out')]

        command_line = (
            'git clone --depth=1 -b $BRANCH --recursive '
            'https://$GITHUB_TOKEN:x-oauth-basic@github.com/$REPO.git _ && '
            'ln -s $AZ_BATCHAI_MOUNT_ROOT/ /data && '
            'ln -s $AZ_BATCHAI_OUTPUT_OUT /out && '
            'cd _ && bash -c {}'
        ).format(shlex.quote(command))

        environment = {
            'REPO': self.git_cfg['repository'],
            'BRANCH': branch,
            'KERAS_BACKEND': 'tensorflow',
        }

        environment_variables = [
            models.EnvironmentVariable(name=k, value=v)
            for k, v in environment.items()]

        secrets = {
            'GITHUB_TOKEN': self.git_cfg['token'],
        }

        try:
            secrets['LOSSWISE_KEY'] = self.cfg['losswise_key']
        except KeyError:
            pass

        secret_variables = [
            models.EnvironmentVariableWithSecretValue(name=k, value=v)
            for k, v in secrets.items()
        ]

        self.client.experiments.create(self.azure_cfg['resource_group'],
                                       self.azure_cfg['workspace_name'],
                                       self.azure_cfg['experiment_name'])

        return models.JobCreateParameters(
            environment_variables=environment_variables,
            secrets=secret_variables,
            cluster=models.ResourceId(id=self._cluster.id),
            node_count=job_node_count,
            container_settings=self._container_settings,
            output_directories=output_directories,
            std_out_err_path_prefix=prefix,
            custom_toolkit_settings=models.CustomToolkitSettings(
                command_line=command_line
            )
        )

    def list(self, state=[]):
        """list jobs.

        Args:

        state (str): comma separated list of states you are interested
          in.

        """
        if isinstance(state, str):
            state = [state]

        states = set(state)
        for job in self.client.jobs.list(models.JobsListOptions()):
            if states and str(job.execution_state).split('.')[1].lower() not in states:
                continue
            print('{name} {ct} {es} {dir}'.format(
                name=job.name,
                ct=job.creation_time,
                es=job.execution_state,
                dir=job.job_output_directory_path_segment,
                cl=job.custom_toolkit_settings.command_line))

    def tensorboard_dir(self, state=[], jobs=[], root=''):
        root = root or self.cfg['local_mount_point']
        interesting = []

        if isinstance(jobs, str):
            jobs = (jobs)

        if jobs and state:
            raise ValueError('Please provide either `state` or `jobs`.')

        if state:
            # FIXME(ryszard): state gets passed as tuple, but jobs is
            # a string... Annoying, don't have time to debug it now.
            if isinstance(state, str):
                state = state.split(',')
            state = set(state)
            for job in self.client.jobs.list(models.JobsListOptions()):
                if states and str(job.execution_state).split('.')[1].lower() not in states:
                    continue
                interesting.append(job)

        elif jobs:
            try:
                jobs = set(jobs.split(','))
            except AttributeError:
                pass
            interesting = [self._get_job(name) for name in jobs]
        else:
            raise ValueError('You have to provide one of `jobs` and `state`.')
        return ','.join('{}:{}'.format(
            job.name,
            os.path.join(
                root,
                job.job_output_directory_path_segment,
                'outputs/out/tensorboard'))
                       for job in interesting)

    def tensorboard(self, state=[], jobs=[], root='', port=6006):
        d = self.tensorboard_dir(state=state, jobs=jobs, root=root)
        os.execlp('tensorboard', 'tensorboard', '--port', str(port),
                  '--logdir', d, '--reload_interval', '1')

    def _get_job(self, job_name):
        return self.client.jobs.get(self.azure_cfg['resource_group'],
                                    self.azure_cfg['workspace_name'],
                                    self.azure_cfg['experiment_name'],
                                    job_name)

    def show(self, job_name):
        """show the yaml-serialized data for the provided job.
        """
        job = self._get_job(job_name)
        print(yaml.dump(job))
        while job.execution_state == models.ExecutionState.queued:
            with blindspin.spinner():
                time.sleep(10)
                job = self.client.jobs.get(self.azure_cfg['resource_group'],
                                           job_name)
                print('execution state:', job.execution_state)
        print(yaml.dump(job))

    def terminate(self, *job_names):
        """terminate the provided jobs.
        """
        for job_name in job_names:
            sys.stdout.write('Terminating {}... '.format(job_name))
            with blindspin.spinner():
                self.client.jobs.terminate(self.azure_cfg['resource_group'],
                                           job_name).wait()
            sys.stdout.write("OK\n")

    def make_job_name(self, length=6):
        return ''.join(random.choice(string.ascii_lowercase + string.digits)
                       for _ in range(length))

    def run(self, job_name=None, command=None, job_node_count=1,
            branch='master'):
        """run `command` on BatchAI.

        Args:
            job_name (str): name of the job. If not provided, random will be
                            chosen.
            command (str): bash command to be run. If empty, it will
                           be read from stdin.
            job_node_count (int): how many nodes to fire up.
            branch (str): branch or commit hash to check out. Command will
                          be run in the contxt of `branch` checked out.
        """
        if not command:
            command = sys.stdin.read()

        if not job_name:
            job_name = self.make_job_name()

        params = self._prepare_job(command, job_node_count, branch, job_name)

        op = self.client.jobs.create(
            self.azure_cfg['resource_group'],
            self.azure_cfg['workspace_name'],
            self.azure_cfg['experiment_name'],
            job_name,
            params
        )
        print(yaml.dump(op.result()))
        print('job_name:', job_name)

    def mount(self, path=''):
        """mount the projects Azure File Storage.

        On Linux, you must be able to do sudo.

        Args:
          path (str): path where to mount.

        """
        path = path or self.cfg['local_mount_point']
        if platform.system() == 'Darwin':
            subprocess.call(['/sbin/mount_smbfs',
                             '//{account}@{account}.file.core.windows.net/{share}'.format(
                                 account=self.azure_cfg['storage_account'],
                                 pwd=self.azure_cfg['storage_key'],
                                 share=self.azure_cfg['azure_file_share'],
                                 cluster=self.azure_cfg['cluster_name']),
                            path])
        else:
            subprocess.call(
                ['sudo', 'mount', '-t', 'cifs',
                 '//{account}.file.core.windows.net/{share}'.format(
                     account=self.azure_cfg['storage_account'],
                     share=self.azure_cfg['azure_file_share']),
                 path,
                 '-o',
                 ','.join(['vers=3.0', 'sec=ntlmssp',
                           'dir_mode=0777', 'file_mode=0777',
                           'username={}'.format(
                               self.azure_cfg['storage_account']),
                           'password={}'.format(
                               self.azure_cfg['storage_key']),
                           'gid={}'.format(os.getgid()),
                           'uid={}'.format(os.getuid())]
                          )])

    def diff_params(self, job_left, job_right):
        root = self.cfg['local_mount_point'] or root
        left = self._get_job(job_left)
        right = self._get_job(job_right)

        files = [
            os.path.join(
                root,
                job.job_output_directory_path_segment,
                'outputs/out/meta.yaml') for job in (left, right)
        ]

        subprocess.call(['diff', *files])

    def params(self, job_name, root=None):
        root = self.cfg['local_mount_point'] or root
        job = self._get_job(job_name)
        with open(os.path.join(
                root,
                job.job_output_directory_path_segment,
                'outputs/out/meta.yaml')) as fi:
            print(fi.read())

    def tail(self, job_name, root='', name='stdout.txt'):
        root = self.cfg['local_mount_point'] or root
        job = self._get_job(job_name)
        path = os.path.join(
            root,
            job.job_output_directory_path_segment,
            'stdouterr',
            name)
        try:
            tail(path, sys.stdout)
        except KeyboardInterrupt:
            pass


def tail(path, stream, sleep=1):
    """naive implementation of tail -f in Python.

    Samba is not very good at update times for files, so `tail -f`
    gets confused. This implementation is inappropriate for big files,
    as it reads the file at `path` every `sleep` seconds in its
    entirety.

    """
    length = 0

    while True:
        with open(path) as fi:
            data = fi.read()
            new_length = len(data)
        if new_length > length:
            stream.write(data[length:])
            length = new_length
            time.sleep(sleep)


if __name__ == '__main__':
    fire.Fire(Job)
