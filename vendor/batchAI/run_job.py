import fire
from azure.storage.file import FileService

import batchaiutils as utils


def run_job(
    subscription_id,
    aad_client_id,
    aad_secret_key,
    aad_tenant_id,
    resource_group,
    location,
    storage_account,
    storage_key,
    job_command,
    container_image=None,
    registry_url=None,
    registry_username=None,
    registry_password=None,
    azure_file_share='afs',
    cluster_name='mit_nc6',
    job_name="test_job",
    job_node_count=1,
    working_dir="poligon",
    output_directory_id="stdouterr",
    stdout_file_name="stdout.txt",
    job_polling_interval=15
):

    cfg = utils.BatchAIConfiguration(
        subscription_id=subscription_id,
        aad_client_id=aad_client_id,
        aad_secret_key=aad_secret_key,
        aad_tenant_id=aad_tenant_id,
        resource_group=resource_group,
        location=location,
        storage_account=storage_account,
        storage_key=storage_key,
        job_command=job_command,
        container_image=container_image,
        registry_url=registry_url,
        registry_password=registry_password,
        registry_username=registry_username,
        azure_file_share=azure_file_share,
        cluster_name=cluster_name,
        job_name=job_name,
        job_node_count=job_node_count,
        working_dir=working_dir,
        output_directory_id=output_directory_id,
        stdout_file_name=stdout_file_name,
        job_polling_interval=job_polling_interval,
    )

    service = FileService(
        cfg.storage_account,
        cfg.storage_key
    )
    service.create_directory(
        cfg.azure_file_share,
        cfg.working_dir,
        fail_on_exist=False
    )

    if "mnist" in cfg.job_command:
        sample_script_url = 'https://raw.githubusercontent.com/fchollet/keras/master/examples/mnist_cnn.py'
        utils.download_file(sample_script_url, 'mnist_cnn.py')

        service.create_file_from_path(
            cfg.azure_file_share,
            cfg.working_dir,
            'mnist_cnn.py',
            'mnist_cnn.py'
        )

    client = utils.create_batchai_client(cfg)
    cluster = utils.get_cluster(cfg, client)
    job_template = utils.prepare_job(cfg, cluster)
    utils.create_job(cfg, client, job_template)
    # utils.wait_for_job_completion(cfg, client)


if __name__ == '__main__':
    fire.Fire(run_job)
