import azure.mgmt.batchai as training
import azure.mgmt.batchai.models as models
from azure.common.credentials import ServicePrincipalCredentials
from azure.storage.file import FileService


class BatchAIConfiguration:
    """
    Configurtion object for Azure Batch AI

    Args:
        subscription_id (str):
            Azure subscription ID
        aad_client_id (str):
            Follow this tutorial:
            https://github.com/Azure/BatchAI/tree/master/recipes#create-credentials-for-service-principal-authentication
        aad_secret_key (str):
            Follow this tutorial:
            https://github.com/Azure/BatchAI/tree/master/recipes#create-credentials-for-service-principal-authentication
        aad_tenant_id (str):
            Follow this tutorial:
            https://github.com/Azure/BatchAI/tree/master/recipes#create-credentials-for-service-principal-authentication
        resource_group (str):
            Azure Resource Group name.
        location (str):
            Location/Region in Azure.
        storage_account (str):
            Azure Storage Account name.
        storage_key (str):
            Azure Storage Account key.
        azure_file_share (:obj:`str`, optional):
            Default is 'afs'.
            Name of Azure File Share mounted
            to Azure Batch AI node.
        admin_name (:obj:`str`, optional):
            Default is 'admin'.
            Admin user name used to setup user
            account on each Azure Batch AI node.
        admin_user_password (:obj:`str`, optional):
            Default is 'P@ssword123'.
            Admin user password used to setup user
            account on each Azure Batch AI node.
            Use instead of 'admin_user_ssh_public_key'.
        admin_user_ssh_public_key (:obj:`str`, optional):
            Default is 'None'.
            Admin user SSH key used to setup user
            account on each Azure Batch AI node.
            Use instead of 'admin_user_password'.
        cluster_name (:obj:`str`, optional):
            Defualt is 'mit_nc6'.
        vm_size (:obj:`str`, optional):
            Default is 'STANDARD_NC6'.
        initial_node_count (:obj:`int`, optional):
            Default is '1'.
            Number of nodes that cluster should
            be initiated with.
        max_node_count (:obj:`int`, optional):
            Default is '4'.
            Maximum number of nodes that can be
            created in Azure Batch AI cluster.
        min_node_count (:obj:`int`, optional):
            Default is '1'.
            Minimum number of nodes that will be kept
            'idle' in Azure Batch AI cluster.
    """

    def __init__(
            self,
            subscription_id,
            aad_client_id,
            aad_secret_key,
            aad_tenant_id,
            resource_group,
            location,
            storage_account,
            storage_key,
            azure_file_share='afs',
            admin_name="admin",
            admin_user_password=None,
            admin_user_ssh_public_key=None,
            cluster_name='mit_nc6',
            vm_size="STANDARD_NC6",
            initial_node_count=0,
            max_node_count=4,
            min_node_count=0,
            workspace_name='batchai-workspace'
    ):
        # common
        self.resource_group = resource_group
        self.location = location
        self.subscription_id = subscription_id
        self.aad_client_id = aad_client_id
        self.aad_secret_key = aad_secret_key
        self.aad_tenant_id = aad_tenant_id
        self.aad_token_uri = \
            'https://login.microsoftonline.com/{0}/oauth2/token' \
                .format(aad_tenant_id)

        # storage
        self.storage_account = storage_account
        self.storage_key = storage_key
        self.azure_file_share = azure_file_share

        # user account settings
        self.admin_name = admin_name
        self.admin_user_password = admin_user_password
        self.admin_user_ssh_public_key = admin_user_ssh_public_key

        # batch AI cluster setup
        self.cluster_name = cluster_name
        self.vm_size = vm_size
        self.initial_node_count = initial_node_count
        self.max_node_count = max_node_count
        self.min_node_count = min_node_count

        self.workspace_name = workspace_name


def create_file_share(cfg):
    """
    Uses 'storage_account', 'storage_key' and 'azure_file_share'
    properties from BatchAIConfiguration object to create a File Share.

    Args:
        cfg (BatchAIConfiguration):
            BatchAIConfiguration object.
    """
    service = FileService(cfg.storage_account, cfg.storage_key)
    service.create_share(cfg.azure_file_share, fail_on_exist=False)
    print('Created File Share folder on Azure Storage Account:\
        {0} with name: {1}'.format(
        cfg.storage_account,
        cfg.azure_file_share))


def prepare_volumes(cfg):
    """
    Returns MountVolumes object based on config

    Args:
        cfg (BatchAIConfiguration):
            BatchAIConfiguration object.
    """
    volumes = models.MountVolumes(
        azure_file_shares=[
            models.AzureFileShareReference(
                account_name=cfg.storage_account,
                credentials=models.AzureStorageCredentialsInfo(
                    account_key=cfg.storage_key
                ),
                azure_file_url='https://{0}.file.core.windows.net/{1}'.format(
                    cfg.storage_account, cfg.azure_file_share
                ),
                relative_mount_path=cfg.azure_file_share)
        ]
    )
    print('Prepared volumes to mount on VM')
    return volumes


def prepare_batchai_template(cfg, volumes):
    """
    Prepares template for BatchAI cluster.
    Returns 'ClusterCreateParameters' object

    Args:
        cfg (BatchAIConfiguration):
            BatchAIConfiguration object.
        volumes (MountVolumes):
            MountVolumes object returned
            by 'prepare_volumes' function.
    """
    parameters = models.ClusterCreateParameters(
        vm_size=cfg.vm_size,
        virtual_machine_configuration=models.VirtualMachineConfiguration(
            image_reference=models.ImageReference(
                publisher="microsoft-ads",
                offer="linux-data-science-vm-ubuntu",
                sku="linuxdsvmubuntu",
                version="latest")),
        scale_settings=models.ScaleSettings(
            auto_scale=models.AutoScaleSettings(
                minimum_node_count=cfg.min_node_count,
                maximum_node_count=cfg.max_node_count,
                initial_node_count=cfg.initial_node_count)
        ),
        node_setup=models.NodeSetup(
            mount_volumes=volumes
        ),
        user_account_settings=models.UserAccountSettings(
            admin_user_name=cfg.admin_name,
            admin_user_password=cfg.admin_user_password,
            admin_user_ssh_public_key=cfg.admin_user_ssh_public_key
        )
    )
    print('Prepared Batch AI template')
    return parameters


def create_batchai_cluster(cfg, batchai_client, template):
    """
    Create a Batch AI cluster using specific template.

    Args:
        cfg (BatchAIConfiguration):
            BatchAIConfiguration object.
        batchai_client (BatchAIManagementClient):
            BatchAIManagementClient object returned
            by 'create_batchai_client' function.
        template (ClusterCreateParameters):
            Batch AI cluster template returned
            by 'prepare_batchai_template' function
    """
    print('Creating Batch AI cluster')
    batchai_client.workspaces.create(cfg.resource_group,
                                     cfg.workspace_name,
                                     cfg.location)

    return batchai_client.clusters.create(
        resource_group_name=cfg.resource_group,
        workspace_name=cfg.workspace_name,
        cluster_name=cfg.cluster_name,
        parameters=template
    ).result()


def get_cluster(cfg, batchai_client):
    """
    Uses 'resource_group' and 'cluster_name' properties
    from BatchAIConfiguration to return specific BatchAI cluster.

    Args:
        cfg (BatchAIConfiguration):
            BatchAIConfiguration object.
        batchai_client (BatchAIManagementClient):
            BatchAIManagementClient object returned
            by 'create_batchai_client' function.
    """
    cluster = batchai_client.clusters.get(cfg.resource_group,
                                          cfg.workspace_name,
                                          cfg.cluster_name)
    return cluster


def print_cluster_status(cluster):
    print(
        '''
        Cluster state: {0}
        ScaleSettings: {1}
        Allocated: {2}
        Idle: {3}
        Unusable: {4}
        Running: {5}
        Preparing: {6}
        Leaving: {7}
        '''.format(
            cluster.allocation_state,
            cluster.scale_settings,
            cluster.current_node_count,
            cluster.node_state_counts.idle_node_count,
            cluster.node_state_counts.unusable_node_count,
            cluster.node_state_counts.running_node_count,
            cluster.node_state_counts.preparing_node_count,
            cluster.node_state_counts.leaving_node_count
        )
    )
    if not cluster.errors:
        return
    for error in cluster.errors:
        print('Cluster error: {0}: {1}'.format(error.code, error.message))
        if error.details:
            print('Details:')
            for detail in error.details:
                print('{0}: {1}'.format(detail.name, detail.value))


def create_batchai_client(configuration):
    """
    Returns BatchAIManagementClient object.

    Args:
        configuration (BatchAIConfiguration):
            BatchAIConfiguration object.
    """
    client = training.BatchAIManagementClient(
        credentials=ServicePrincipalCredentials(
            client_id=configuration.aad_client_id,
            secret=configuration.aad_secret_key,
            token_uri=configuration.aad_token_uri
        ),
        subscription_id=configuration.subscription_id
    )
    print('Created BatchAI client')
    return client
