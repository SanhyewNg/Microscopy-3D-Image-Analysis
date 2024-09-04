import fire

import vendor.batchAI.batchaiutils as utils


def create_cluster(
    subscription_id,
    aad_client_id,
    aad_secret_key,
    aad_tenant_id,
    resource_group,
    location,
    storage_account,
    storage_key,
    azure_file_share='afs',
    admin_name="mituser",
    admin_user_password="",
    admin_user_ssh_public_key="",
    cluster_name='mit_nc6',
    vm_size="STANDARD_NC6",
    initial_node_count=0,
    max_node_count=4,
    min_node_count=0,
    workspace_name='batchai-workspace'
):

    batchai_cfg = utils.BatchAIConfiguration(
        subscription_id=subscription_id,
        aad_client_id=aad_client_id,
        aad_secret_key=aad_secret_key,
        aad_tenant_id=aad_tenant_id,
        resource_group=resource_group,
        location=location,
        storage_account=storage_account,
        storage_key=storage_key,
        azure_file_share=azure_file_share,
        admin_name=admin_name,
        admin_user_password=admin_user_password,
        admin_user_ssh_public_key=admin_user_ssh_public_key,
        cluster_name=cluster_name,
        vm_size=vm_size,
        initial_node_count=initial_node_count,
        max_node_count=max_node_count,
        min_node_count=min_node_count,
        workspace_name=workspace_name
    )

    utils.create_file_share(batchai_cfg)
    volumes = utils.prepare_volumes(batchai_cfg)
    template = utils.prepare_batchai_template(batchai_cfg, volumes)
    client = utils.create_batchai_client(batchai_cfg)
    utils.create_batchai_cluster(batchai_cfg, client, template)
    cluster = utils.get_cluster(batchai_cfg, client)
    utils.print_cluster_status(cluster)


if __name__ == '__main__':
    fire.Fire(create_cluster)
