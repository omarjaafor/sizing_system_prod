#!/usr/bin/env python
# coding: utf-8

import os
import time
import urllib

from lib import plogger

plogger.create_logger("storage-errors")


def decompose_gs_path(gs_path):
    """
    Function to separate the bucket and the file name from
    the full path file in GCS

        Parameters
        ----------
        gs_path: str
            the full path of a blob in a GCS bucket

        Returns
        -------
        str, str:
            bucket-name and path/to/gs/blob
    """
    bucket = gs_path.replace("gs://", "").split("/")[0]
    file_path = gs_path[len(bucket) + 6 :]
    return bucket, file_path


def upload_blob(
    storage_client,
    bucket_name,
    source_file_name,
    destination_blob_name,
    *,
    overwrite=False,
    logger_name=None
):
    """
    Uploads a file to the GCS bucket

    Parameters
    ----------
    storage_client: `google.cloud.storage.Client`,
        GCP storage client
    bucket_name: str,
        your bucket name
    source_file_name: str,
        local/path/to/file
    destination_blob_name: str,
        storage object name
    overwrite: bool, optional, default: False
        Overwrite existing blob if True
    logger_name: str, optional, default: None,
        The name of the logger to be used.

    Returns
    -------
    bool:
        True if upload succeed, otherwise false
    """
    labels = {
        "source_file_name": source_file_name,
        "bucket_name": bucket_name,
        "destination_blob_name": destination_blob_name,
    }
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        if overwrite or not blob.exists():
            blob.upload_from_filename(source_file_name)
            message = "SUCCEED upload {} >> {}".format(
                source_file_name, destination_blob_name
            )
            message = plogger.create_log_message(
                message, __file__, upload_blob.__name__, "INFO"
            )
            plogger.log(
                message=message, severity="INFO", labels=labels, logger_name=logger_name
            )

    except Exception as e:
        message = "FAILED upload {} >> {} - {}".format(
            source_file_name, destination_blob_name, e
        )
        message = plogger.create_log_message(
            message, __file__, upload_blob.__name__, "WARNING"
        )
        plogger.log(
            message=message, severity="WARNING", labels=labels, logger_name=logger_name
        )
        return False

    return True


def upload_file_to_gs(storage_client, local_path, gs_path, logger_name=None):
    """
    Function to upload a file, locally stored, on GCS

        Parameters
        ----------
        storage_client : `google.cloud.storage.Client`,
            GCP storage client
        local_path : str,
            local path of the file (local/path/to/file)
        gs_path: str,
            full gs output path file
        logger_name: str, optional, default: None,
            The name of the logger to be used.
        Returns
        -------
        bool
            True if the file has been uploaded, False otherwise
    """
    bucket_name, file_name = decompose_gs_path(gs_path)
    labels = {
        "source_file_name": local_path,
        "bucket_name": bucket_name,
        "destination_blob_name": file_name,
    }
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(local_path)

        message = "SUCCEED UPLOAD {} >> {}".format(local_path, gs_path)
        message = plogger.create_log_message(
            message, __file__, upload_file_to_gs.__name__, "INFO"
        )
        plogger.log(message, severity="INFO", labels=labels, logger_name=logger_name)
        return True
    except Exception as e:
        message = "FAILED upload {} >> {} - {}".format(local_path, gs_path, e)
        message = plogger.create_log_message(
            message, __file__, upload_file_to_gs.__name__, "WARNING"
        )
        plogger.log(
            message=message, severity="WARNING", labels=labels, logger_name=logger_name
        )


def transfer_url_to_gs(storage_client, source_url, gs_path, *, logger_name=None):
    """
    Function to transfer a file, from URL, to GCS

    Parameters
    ----------
    storage_client: `google.cloud.storage.Client`,
        GCP storage client
    source_url: str
        url of the file
    gs_path: str
        full gs output path file
    logger_name: str, optional, default: None
        the name of the logger to be used
    Returns
    -------
    bool
        True if the file has been transfered, False otherwise
    """
    try:
        bucketName, fileName = decompose_gs_path(gs_path)
        bucket = storage_client.get_bucket(bucketName)
        blob = bucket.blob(fileName)
        if blob.exists():
            return True

        with urllib.request.urlopen(source_url) as rqst:
            blob.upload_from_string(rqst.read())

        message = "SUCCEED transfer {} >> {}".format(source_url, gs_path)
        message = plogger.create_log_message(
            message, __file__, transfer_url_to_gs.__name__, "INFO"
        )
        plogger.log(message=message, severity="INFO", logger_name=logger_name)
        return True
    except Exception as e:
        message = "FAILED transfer {} >> {} - {}".format(source_url, gs_path, e)
        message = plogger.create_log_message(
            message, __file__, transfer_url_to_gs.__name__, "WARNING"
        )
        plogger.log(message=message, severity="WARNING", logger_name=logger_name)


def blob_exists(storage_client, gs_path, logger_name=None):
    """
    Check whether a blob exists on GCS

    Parameters
    ----------
    storage_client: `google.cloud.storage.Client`,
        GCP storage client
    gs_path: str
        full gs output path file
    logger_name: str, optional, default: None,
        The name of the logger to be used.
    Returns
    -------
    bool
        True if the blob exists, False otherwise
    """
    try:
        bucketName, fileName = decompose_gs_path(gs_path)

        bucket = storage_client.get_bucket(bucketName)
        return bucket.blob(fileName).exists()
    except Exception as e:
        message = plogger.create_log_message(
            "{}".format(e), __file__, blob_exists.__name__, "WARNING"
        )
        plogger.log(message=message, severity="WARNING", logger_name=logger_name)
        return False


def download_blob(
    storage_client,
    bucket_name,
    source_blob_name,
    destination_file_name,
    logger_name=None,
):
    """
    Downloads a blob from the bucket.

        Parameters
        ----------
        storage_client: `google.cloud.storage.Client`,
            GCP storage client
        bucket_name: str,
            your bucket name
        source_blob_name: str,
            storage-object-name
        destination_file_name: str,
            local/path/to/file
        logger_name: str, optional, default: None,
            The name of the logger to be used.
        Returns
        -------
        bool
            True if the file has been downloaded
        Raises
        ------
        Exception
            if download fails; catch and re-raise any occurred exception
    """
    labels = {
        "source_blob_name": source_blob_name,
        "bucket_name": bucket_name,
        "destination_file_name": destination_file_name,
    }
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        message = "SUCCEED download {} >> {}".format(
            source_blob_name, destination_file_name
        )
        plogger.log(message=message, severity="INFO", logger_name=logger_name)
    except Exception as e:
        message = "FAILED download {} >> {} - {}".format(
            source_blob_name, destination_file_name, e
        )
        message = plogger.create_log_message(
            message, __file__, download_blob.__name__, "ERROR"
        )
        plogger.log(
            message=message, severity="ERROR", labels=labels, logger_name=logger_name
        )
        raise

    return True


def download_file_from_gs(storage_client, gsPath, localPath, retry=0):
    """
    Function to download locally a file from GCS

        Parameters
        ----------
        storage_client:
            GCP storage client
        gsPath: str
            full gs input path file
        localPath: str
            desired local path of the file
        retry: int
    """
    try:
        bucketName, fileName = decompose_gs_path(gsPath)
        bucket = storage_client.get_bucket(bucketName)
        blob = bucket.get_blob(fileName)
        blob.download_to_filename(localPath)
        # logger.log_text("File saved locally from GS :: gsPath {}".format(localPath), severity='INFO')
        print("File saved locally from GS :: gsPath {}".format(localPath))
        return True
    except Exception as e:
        if retry < 3:
            # logger.log_text("Error in downloading file from GS :: localPath {} :: gsPath {} :: retry {}".format(
            # localPath,gsPath,retry+1), severity='ERROR')
            print(
                "Error in downloading file from GS :: localPath {} :: gsPath {} :: retry {}".format(
                    localPath, gsPath, retry + 1
                )
            )
            time.sleep(3)
            return download_file_from_gs(
                storage_client, gsPath, localPath, retry=retry + 1
            )
        else:
            # logger.log_text("Error in downloading file from GS :: gsPath {} :: {}. Max number of retry
            # exceeded".format(gsPath,e), severity='ERROR')
            print(
                "Error in downloading file from GS :: gsPath {} :: {}. Max number of retry exceeded".format(
                    gsPath, e
                )
            )
            return False


# prefix: str, optional, default: ''
# prefix used to filter blobs.


def create_local_directory(local_path):
    if local_path is None or local_path == "":
        return

    if not os.path.exists(local_path):
        os.makedirs(local_path)


def download_blobs_from_bucket_dir(
    bucket, bucket_directory, local_directory, recursive, logger_name
):
    labels = {
        "bucket_directory": bucket.name + "/" + bucket_directory,
        "local_directory": local_directory,
    }
    message = "START download {}/{} >> {}".format(
        bucket.name, bucket_directory, local_directory
    )
    message = plogger.create_log_message(
        message, __file__, download_blobs_from_bucket_dir.__name__, severity="INFO"
    )
    plogger.log(
        message=message, severity="INFO", labels=labels, logger_name=logger_name
    )

    local_path = os.path.join(local_directory, bucket_directory)
    create_local_directory(local_path)

    iterator = bucket.list_blobs(versions=True, prefix=bucket_directory, delimiter="/")

    subdirectories = iterator.prefixes
    blobs = list(iterator)

    for blob in blobs:
        local_name = blob.name.split("/")[-1]
        if local_name == "":
            continue
        blob.download_to_filename(os.path.join(local_path, local_name))

    message = "END download {}/{} >> {}".format(
        bucket.name, bucket_directory, local_directory
    )
    message = plogger.create_log_message(
        message, __file__, download_blobs_from_bucket_dir.__name__, severity="INFO"
    )
    plogger.log(
        message=message, severity="INFO", labels=labels, logger_name=logger_name
    )

    if not recursive:
        return

    for sub_directory in subdirectories:
        download_blobs_from_bucket_dir(
            bucket, sub_directory, local_directory, recursive, logger_name
        )


def download_blobs(
    storage_client,
    bucket_name,
    bucket_directory=None,
    local_directory=None,
    recursive=False,
    logger_name=None,
):
    """
    Download a bucket directory to a local directory with recursive ability.

    Parameters
    ----------
    storage_client: `google.cloud.storage.Client`,
            GCP storage client
    bucket_name: str,
        your bucket name
    bucket_directory: str, optional, default: None
        bucket directory, None or empty means root directory
    local_directory: str,
        local/path/to/directory where blobs are to be downloaded
    recursive:
        if true, downloads all blobs to local dir and sub_dirs.
    logger_name: str, optional, default: None,
        The name of the logger to be used.
    """
    bucket = storage_client.get_bucket(bucket_name)

    if local_directory is None:
        local_directory = "."
    if bucket_directory is None:
        bucket_directory = ""

    bucket_directory = bucket_directory.strip(".").strip("/")
    if bucket_directory != "":
        bucket_directory = bucket_directory + "/"

    try:
        download_blobs_from_bucket_dir(
            bucket, bucket_directory, local_directory, recursive, logger_name
        )
    except Exception as e:
        message = "FAILED download {} >> {} - {}".format(
            bucket_name, local_directory, e
        )
        message = plogger.create_log_message(
            message, __file__, download_blobs.__name__, "ERROR"
        )
        plogger.log(message=message, severity="ERROR", logger_name=logger_name)
        raise


def get_list_local_files(local_path, recursive=False):
    """
    Create a list of file full paths in a local folder.

        Parameters
        ----------
        local_path: str,
            path/to/local/dir
        recursive: bool, optional, default: False
            if True, includes files in sub directories

        Returns
        -------
        list[str]
            a list of file full paths in a local folder.
    """
    # create a list of file and sub directories
    # names in the given directory
    list_files = os.listdir(local_path)
    all_files = list()
    # Iterate over all the entries
    for entry in list_files:
        # Create full path
        full_path = os.path.join(local_path, entry)
        # If entry is a directory then get the list of files in this directory
        if recursive and os.path.isdir(full_path):
            all_files = all_files + get_list_local_files(full_path)
        elif not os.path.isdir(full_path):
            all_files.append(full_path)

    return all_files


def fix_path_for_gs(local_file_path):
    """
    Remove point and slash from the beginning of a file path

    Parameters
    ----------
    local_file_path: str,
        local file paths './path/to/local/file'

    Returns
    -------
    str
        path without './' at the beginning path/to/local/file
    """

    return local_file_path.strip(".").strip("/")


def upload_blobs(
    storage_client,
    bucket_name,
    list_local_paths,
    destination_bucket_directory,
    *,
    keep_structure=False,
    overwrite=False,
    logger_name=None
):
    """
    Uploads a list of local files to a GCS bucket

        Parameters
        ----------
        storage_client: `google.cloud.storage.Client`,
            GCP storage client
        bucket_name: str,
            your bucket name
        list_local_paths: list[str],
            list of local/path/to/file
        destination_bucket_directory: str,
            path/to/dir
        keep_structure: bool, optional, default: False,
            Tells whether to keep directory structures in the bucket
        overwrite: bool, optional, default: False
            Overwrite existing blob if True
        logger_name: str, optional, default: None,
            The name of the logger to be used.

        Returns
        -------
        bool:
            True if all files uploaded, otherwise false
    """
    bucket = storage_client.bucket(bucket_name)
    destination_bucket_directory = fix_path_for_gs(destination_bucket_directory)
    for local_path in list_local_paths:
        if keep_structure:
            destination_blob_name = (
                destination_bucket_directory + "/" + fix_path_for_gs(local_path)
            )
        else:
            destination_blob_name = (
                destination_bucket_directory + "/" + os.path.basename(local_path)
            )

        labels = {
            "local_file_name": local_path,
            "bucket_name": bucket_name,
            "destination_blob_name": destination_blob_name,
        }
        try:
            blob = bucket.blob(destination_blob_name)
            if overwrite or not blob.exists():
                blob.upload_from_filename(local_path)

                message = "SUCCEED upload {} >> {}".format(
                    local_path, destination_blob_name
                )
                message = plogger.create_log_message(
                    message, __file__, upload_blobs.__name__, "INFO"
                )
                plogger.log(
                    message=message,
                    severity="INFO",
                    labels=labels,
                    logger_name=logger_name,
                )
        except Exception as e:
            message = "FAILED upload {} >> {} - {}".format(
                local_path, destination_blob_name, e
            )
            message = plogger.create_log_message(
                message, __file__, download_blob.__name__, "WARNING"
            )
            plogger.log(
                message=message,
                severity="WARNING",
                labels=labels,
                logger_name=logger_name,
            )

    return True
