# coding: utf-8
import threading
import atexit
import sys
from flask import Flask

# from flask_socketio import SocketIO
import traceback
from gevent.pywsgi import WSGIServer
import os
import os.path
from google.cloud import logging as gcl
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import pubsub_v1
from flask import Flask, request, abort, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import time
import uuid
import logging
import os
import subprocess
import re
import json
from flask import (
    jsonify,
    request,
    send_file,
    send_from_directory,
    session,
    Response,
    make_response,
)
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
import datetime
from dateutil import tz
from time import strftime, localtime
import logging
from google.cloud import storage
from google.cloud import pubsub_v1

# from date import get_modification_date
import json
import pandas as pd

# from errors import InvalidRequest
# from utils_interface import (dump_settings)
import time
from datetime import datetime
from google.cloud import logging as gcl
from google.cloud.logging.handlers import CloudLoggingHandler
import tempfile
from multiprocessing import Pool, Manager
import subprocess
import argparse
import logging
import pred_mensuration
from google.cloud import pubsub_v1
from lib.gcp_function import (
    delete_input_file,
    delete_run_id,
    get_list_local_files,
    upload_blobs,
    download_blob,
    upload_blob,
    download_blobs,
    copy_blob,
    extractBucketFile,
    extract_date_run_name_from_run_id,
)

from werkzeug.utils import secure_filename

import pred_morphotype
from lib.constants import *
from lib import plogger
from lib.server import CamaieuApplicationServer
import datetime
from lib.utils import *
import zipfile


def gen_run_id(run_name):
    time_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_name = re.sub(r"[^-_A-Za-z0-9]+", r"_", run_name).lower()
    run_id = time_tag + "_" + run_name
    run_id = run_id.replace("_", "-")
    return run_id


publisher = pubsub_v1.PublisherClient()

global manager
manager = Manager()

plogger.create_logger(os.getenv(LOGGER_NAME, LOGGER_NAME_VAL))
plogger.log("Activating debugs for flask CORS")
logging.getLogger("flask_cors").level = logging.DEBUG

# import EXE
# import GBSS
pref = "/"
# pref="../../../../../"
# a supprimer

user_settings = pref + "input/sizing-system/setting/params.json"
user_settings_GBSS = pref + "input/sizing-system/setting/params_GBSS.json"


def publish(publisher_client, project_id, topic_name, message, retry=0):
    """
    Function to publish a message in PubSub topic
    Input :
        publisher_client : GCP PubSub publisher client
        project_id : id of the GCP project
        topic_name : name of the topic where the message will be published
        message : message to publish (str)
    Output :
        True if the message has been published, False otherwise
    """
    topic_path = publisher_client.topic_path(project_id, topic_name)
    publisher_client.get_topic(topic_path)
    msg = plogger.create_log_message(
        "START publish message:{}".format(message), __file__, publish.__name__, "INFO"
    )
    plogger.log(msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
    try:
        publisher_client.publish(topic_path, data=message)
        msg = "SUCCESSFUL Publishing - message:{}".format(message)
        msg = plogger.create_log_message(msg, __file__, publish.__name__, "INFO")
        plogger.log(message=msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

        return True
    except Exception as e:
        msg = "FAILED publisher flask_app - message:{},error:{}".format(e, message)
        msg = plogger.create_log_message(message, __file__, publish.__name__, "WARNING")
        plogger.log(message=msg, severity="WARNING", logger_name=LOGGER_NAME_VAL)
        # Should work on 2nd try
        if retry < 2:
            return publish(publisher, project_id, topic_name, message, retry=retry + 1)
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()
            trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
            log_msg = "FAILED Publisher - SEVERE - {} - {}".format(e, trace_back)
            log_msg = plogger.create_log_message(log_msg, __file__, "__main__", "ERROR")
            plogger.log(message=log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
            return False


client = gcl.Client()
app = CamaieuApplicationServer(__name__)
CORS(app)
GCS_DATA_FOLDER = os.getenv("GCS_DATA_FOLDER", "syzingsys")
# upload_dir="/data/upload"
global env_
env_ = manager.dict()
# Clients storage and pubsub
storage_client = storage.Client()
publisher = pubsub_v1.PublisherClient()

# Universe file
session = {}

POOL_TIME = 5
commonDataStruc = {}
dataLock = threading.Lock()
yourThread = threading.Thread()


# socketio = SocketIO(app)
def create_app(json_, sess_id, env_, BMI, Age, quadrillage):
    log_params = {
        "json_": json_,
        "sess_id": sess_id,
        "BMI": BMI,
        "Age": Age,
        "quadrillage": quadrillage,
    }
    log_msg = "START create_app({params})".format(params=log_params)
    log_msg = plogger.create_log_message(log_msg, __file__, create_app.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    js = json_
    js["sess_id"] = sess_id
    js["BMI"] = BMI
    js["Age"] = Age
    js["quadrillage"] = quadrillage

    try:
        json_file = pref + "input/sizing-system/setting/input_" + sess_id + ".json"
        with open(json_file, "w") as fp:
            json.dump(js, fp)
        os.system(
            "gsutil cp " + json_file + " " + bucket + "input/sizing-system/setting/"
        )
        # upload_blob(storage_client, BUCKET_NAME, json_file, "input/sizing-system/setting/input_"+sess_id+".json")
    except Exception as e:
        log_msg = "FAILED to read {} - {}".format(json_file, e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, create_app.__name__, "ERROR"
        )
        plogger.log(message=log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise

    def interrupt():
        log_msg = "INTERRUPT thread"
        log_msg = plogger.create_log_message(
            log_msg, __file__, interrupt.__name__, "INFO"
        )
        plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
        global yourThread
        yourThread.cancel()

    def EXE_gentaillant():
        script = ["python", "launcher.py", "--run_id=" + sess_id]
        try:  # temporary FIX added waiting to remove the final error in PREVINIT
            # subprocess.run(script, check=True)
            msg = {"kind": KIND, "name": sess_id}
            message_json = json.dumps(msg)
            enc_message = message_json.encode("utf-8")
            # publish(publisher, PROJECT_ID, TOPIC_NAME, enc_message, retry=0)
            os.system("python launcher.py --run_id=" + sess_id)
        except Exception as e:
            log_msg = "FAILED - {}".format(e)
            log_msg = plogger.create_log_message(
                log_msg, __file__, create_app.__name__, "ERROR"
            )
            plogger.log(message=log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
            raise

    def launcher():
        global yourThread
        yourThread = threading.Timer(POOL_TIME, EXE_gentaillant, ())
        yourThread.start()

    # Initiate
    launcher()
    # When you kill Flask (SIGTERM), clear the trigger for the next thread
    atexit.register(interrupt)


ALLOWED_EXTENSIONS = set(["txt", "dat", "xlsx", "csv", "xls"])


def allowed_file(filename):
    if (
        "genetique" in filename.lower()
        or "quadrillage" in filename.lower()
        or "morpho" in filename.lower()
        or "pred_mens" in filename.lower()
    ):
        return False
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/download_file", methods=["GET"])
def download_files():
    plogger.log(">> GET /api/list_results_files2", severity="INFO")
    plogger.log(">> Listing previous batch runs", severity="INFO")

    try:
        file_name = request.args.get("file_name")
    except:
        log_msg = "Failed to read file name"
        log_msg = plogger.create_log_message(
            log_msg, __file__, download_files.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Indiquez le file_name"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        folder_ = request.args.get("folder")
    except:
        log_msg = "Failed to retrieve folder, empty folder"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Indiquez le dossier"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
    if folder_ != "morpho" and folder_ != "pred_mens":
        log_msg = "Failed to retrieve folder, does not exist"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Pas de dossier avec ce nom"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    if folder_ == "morpho":
        main_output_dir = "/output/sizing-system/result/morpho/batch/"  # os.path.join('gs://', GCS_DATA_FOLDER, RESULTS_PATH_REL)
    elif folder_ == "pred_mens":
        main_output_dir = "/output/sizing-system/result/pred_mens/batch/"
        plogger.log(main_output_dir)
    local_path = main_output_dir + file_name
    if os.path.isfile(local_path):
        return send_file(local_path)
    else:
        log_msg = "Failed to retrieve file, does not exist"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Pas de fichier local avec ce nom"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400


@app.route("/list_results_files2", methods=["GET"])
def list_results_files2():
    plogger.log(">> GET /api/list_results_files2", severity="INFO")
    plogger.log(">> Listing previous batch runs", severity="INFO")
    try:
        folder_ = request.args.get("folder")
    except:
        log_msg = "Failed to retrieve folder, empty folder"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Indiquez le dossier"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
    if (
        folder_ != "morpho"
        and folder_ != "pred_mens"
        and folder_ != "quadrillage"
        and folder_ != "genetique"
    ):
        log_msg = "Failed to retrieve folder, does not exist"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Pas de dossier avec ce nom"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    if folder_ == "morpho":
        main_output_dir = "gs://smart-fashion-data/output/sizing-system/result/morpho/batch/"  # os.path.join('gs://', GCS_DATA_FOLDER, RESULTS_PATH_REL)
    elif folder_ == "pred_mens":
        main_output_dir = (
            "gs://smart-fashion-data/output/sizing-system/result/pred_mens/batch/"
        )
    elif folder_ == "quadrillage":
        main_output_dir = (
            "gs://smart-fashion-data/output/sizing-system/result/quadrillage/zip"
        )
    elif folder_ == "genetique":
        main_output_dir = (
            "gs://smart-fashion-data/output/sizing-system/result/genetique/zip"
        )

    list_dirs = list_dirs_from_gs(storage_client, main_output_dir)
    list_dirs.sort(reverse=True)
    log_msg = plogger.create_log_message(
        str(list_dirs), __file__, list_results_files2.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
    plogger.log(">> Extracting informations from existing directories on gs")
    result_dirs = []
    for run_id in list_dirs:
        try:
            date, run_name, gs_filepath = extract_date_run_name_from_run_id(
                run_id, folder_, plogger
            )
            result_dirs.append(
                {
                    "date": date,
                    "full_name": run_id,
                    "gs_filepath": gs_filepath,
                    "run_id": run_id.split("_")[0],
                }
            )
        except:
            None
    plogger.log(">> Returning response")
    return jsonify(result_dirs)


@app.route("/list_input_files", methods=["GET"])
def list_input_files():
    plogger.log(">> GET /api/list_results_files2", severity="INFO")
    plogger.log(">> Listing previous batch runs", severity="INFO")
    try:
        folder_ = request.args.get("folder")
    except:
        log_msg = "Failed to retrieve folder, empty folder"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Indiquez le dossier"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
    if (
        folder_ != "morpho"
        and folder_ != "pred_mens"
        and folder_ != "quadrillage"
        and folder_ != "genetique"
    ):
        log_msg = "Failed to retrieve folder, does not exist"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Pas de dossier avec ce nom"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    if folder_ == "morpho":
        main_output_dir = "gs://smart-fashion-data/input/sizing-system/data/"  # os.path.join('gs://', GCS_DATA_FOLDER, RESULTS_PATH_REL)
    elif folder_ == "pred_mens":
        main_output_dir = "gs://smart-fashion-data/input/sizing-system/data/"
    elif folder_ == "quadrillage":
        main_output_dir = "gs://smart-fashion-data/input/sizing-system/data/"
    elif folder_ == "genetique":
        main_output_dir = "gs://smart-fashion-data/input/sizing-system/data/"

    list_dirs_all = list_dirs_from_gs(storage_client, main_output_dir)
    list_dirs = []
    plogger.log(
        plogger.create_log_message(str(list_dirs_all), "INFO"),
        logger_name=LOGGER_NAME_VAL,
    )
    for l in list_dirs_all:
        plogger.log(
            plogger.create_log_message(str(l) + "_" + str(folder_ + "_" in l), "INFO"),
            logger_name=LOGGER_NAME_VAL,
        )
        if folder_ + "_" in l:
            list_dirs.append(l)
    list_dirs.sort(reverse=True)

    log_msg = plogger.create_log_message(
        str(list_dirs), __file__, list_results_files2.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
    plogger.log(">> Extracting informations from existing directories on gs")
    result_dirs = []
    for run_id in list_dirs:
        try:
            date, run_name, gs_filepath = extract_date_run_name_from_run_id(
                run_id, folder_, plogger
            )
            result_dirs.append(
                {
                    "date": date,
                    "full_name": run_id,
                    "gs_filepath": "gs://smart-fashion-data/input/sizing-system/data/",
                    "run_id": run_id.split("_")[0],
                }
            )
        except:
            None
    plogger.log(">> Returning response")
    return jsonify(result_dirs)


@app.route("/delete_run", methods=["GET"])
def delete_run():
    plogger.log(">> GET /api/delete_run", severity="INFO")

    try:
        plogger.log(">> Delete existing run from run id")
        plogger.log(">> Extracting params from request")
        try:
            run_id = request.args.get("run_id")
            if run_id in [None, 0]:
                raise Exception("BAD_JSON: missing run id in arguments")
        except:
            raise Exception("BAD_JSON: missing run id in arguments")
        delete_run_id(run_id, plogger)
        d = {"success": True, "run_id": run_id}
        plogger.log(">> SUCCESS")
    except:
        e = traceback.format_exc()
        plogger.log(">> FAILED", severity="ERROR")
        plogger.log(e, severity="ERROR")
        error_code = 500
        response = make_response(jsonify(error=e, success=False), error_code)
        abort(response)
    plogger.log(">> Preparing response")
    res = json.dumps(d, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    plogger.log(">> Returning response")
    return resp


@app.route("/delete_input_file", methods=["GET"])
def delete_file():
    plogger.log(">> GET /api/delete_input_file", severity="INFO")

    try:
        plogger.log(">> Delete existing input file")
        plogger.log(">> Extracting params from request")
        try:
            filename = request.args.get("filename")
            if filename in [None, 0]:
                raise Exception("BAD_JSON: missing filename in arguments")
        except:
            raise Exception("BAD_JSON: missing filename in arguments")
        delete_input_file(filename, plogger)
        d = {"success": True, "filename": filename}
        plogger.log(">> SUCCESS")
    except:
        e = traceback.format_exc()
        plogger.log(">> FAILED", severity="ERROR")
        plogger.log(e, severity="ERROR")
        error_code = 500
        response = make_response(jsonify(error=e, success=False), error_code)
        abort(response)
    plogger.log(">> Preparing response")
    res = json.dumps(d, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    plogger.log(">> Returning response")
    return resp


@app.route("/delete_multiple_files", methods=["GET"])
def delete_multiple_files():
    plogger.log(">> GET /api/delete_multiple_files", severity="INFO")
    try:
        plogger.log(">> Delete existing files from filename")
        plogger.log(">> Extracting params from request")
        try:
            filename_list = request.args.get("filename_list").split(",")
            if filename_list in [None, 0]:
                raise Exception("BAD_JSON: missing filename_list in arguments")
        except:
            raise Exception("BAD_JSON: missing run id list in arguments")
        for filename in filename_list:
            delete_input_file(filename, plogger)
        d = {"success": True, "filename_list": filename_list}
        plogger.log(">> SUCCESS")
    except:
        e = traceback.format_exc()
        plogger.log(">> FAILED", severity="ERROR")
        plogger.log(e, severity="ERROR")
        error_code = 500
        response = make_response(jsonify(error=e, success=False), error_code)
        abort(response)
    plogger.log(">> Preparing response")
    res = json.dumps(d, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    plogger.log(">> Returning response")
    return resp


@app.route("/delete_multiple_runs", methods=["GET"])
def delete_multiple_runs():
    plogger.log(">> GET /api/delete_multiple_runs", severity="INFO")
    try:
        plogger.log(">> Delete existing runs from run id")
        plogger.log(">> Extracting params from request")
        try:
            run_id_list = request.args.get("run_id_list").split(",")
            if run_id_list in [None, 0]:
                raise Exception("BAD_JSON: missing run id list in arguments")
        except:
            raise Exception("BAD_JSON: missing run id list in arguments")
        for run_id in run_id_list:
            delete_run_id(run_id, plogger)
        d = {"success": True, "run_id_list": run_id_list}
        plogger.log(">> SUCCESS")
    except:
        e = traceback.format_exc()
        plogger.log(">> FAILED", severity="ERROR")
        plogger.log(e, severity="ERROR")
        error_code = 500
        response = make_response(jsonify(error=e, success=False), error_code)
        abort(response)
    plogger.log(">> Preparing response")
    res = json.dumps(d, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    plogger.log(">> Returning response")
    return resp


def list_dirs_from_gs(storage_client, gsPath):
    """
    Function to list folders into GCS bucket
    Input :
        storage_client : GCP storage client
        gsPath : full GS path
    Output :
        list of folders in gsPath
    """
    try:
        if gsPath[-1] != "/":
            gsPath = gsPath + "/"
        bucketName, fileName = extractBucketFile(gsPath)
        bucket = storage_client.get_bucket(bucketName)
        return list(
            set(
                [
                    f.name[len(fileName) :].split("/")[0]
                    for f in bucket.list_blobs(prefix=fileName)
                ]
            )
        )

    except Exception as e:
        plogger.log(
            "Error in list dirs from GS :: gsPath {} :: {}".format(gsPath, e),
            severity="ERROR",
        )
        return None


def upload_file_helper(request, SESSION_ID):
    try:
        folder_ = request.args.get("folder")
    except:
        log_msg = "Failed to retrieve folder, empty folder"
        log_msg = plogger.create_log_message(
            log_msg, __file__, list_results_files2.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": "Indiquez le dossier"}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            log_msg = "FAILED to load file: Empty File"
            log_msg = plogger.create_log_message(
                log_msg, __file__, gen_morpho.__name__, "ERROR"
            )
            plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
            out = {"success": False, "exception": log_msg}
            res = json.dumps(out, indent=4)
            resp = Response(res)
            resp.headers["Content-Type"] = "application/json"
            return resp, 400
        file = request.files["file"]
        if file.filename == "":
            log_msg = "FAILED to load file: Empty Filename"
            log_msg = plogger.create_log_message(
                log_msg, __file__, gen_morpho.__name__, "ERROR"
            )
            plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
            out = {"success": False, "exception": log_msg}
            res = json.dumps(out, indent=4)
            resp = Response(res)
            resp.headers["Content-Type"] = "application/json"
            return resp, 400
        if file and allowed_file(file.filename):
            filename = SESSION_ID + "_" + folder_ + "_" + secure_filename(file.filename)
            file.save(os.path.join("/input/sizing-system/data/", filename))
            os.system(
                "gsutil cp /input/sizing-system/data/"
                + filename
                + " "
                + bucket
                + "input/sizing-system/data/"
            )

            # flash('File successfully uploaded')
            log_msg = "Loaded file {} successfully".format(filename)
            log_msg = plogger.create_log_message(
                log_msg, __file__, gen_morpho.__name__, "INFO"
            )
            plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
            out = {"success": True, "run_id": SESSION_ID, "filename": filename}
            res = json.dumps(out, indent=4)
            resp = Response(res)
            resp.headers["Content-Type"] = "application/json"
            return resp, 200

        else:
            # flash('Allowed file types are txt, csv, xlsx')
            log_msg = "FAILED to load file: accepted types: csv, txt, xls, xlsx, dat"
            log_msg = plogger.create_log_message(
                log_msg, __file__, gen_morpho.__name__, "ERROR"
            )
            plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)

            out = {"success": False, "exception": log_msg}
            res = json.dumps(out, indent=4)
            resp = Response(res)
            resp.headers["Content-Type"] = "application/json"
            return resp, 400


@app.route("/upload", methods=["POST"])
def upload_file():
    print("PRINTING REQ FILES")
    print(request.files)
    SESSION_ID = gen_run_id(request.args.get("SESSION_ID"))
    return upload_file_helper(request, SESSION_ID)


@app.route("/pred_taillant/morpho", methods=["POST"])
def gen_morpho():
    log_msg = ">> POST /pred_taillant/morpho"
    log_msg = plogger.create_log_message(log_msg, __file__, gen_morpho.__name__, "INFO")
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    mesures = None
    try:
        # 'Stature (mm)', 'Hip Circumference, Maximum (mm)','Chest Circumference (mm)', 'Waist Circumference, Pref (mm)','BustChest Circumference Under Bust (m$
        mesures = [c for c in request.args.get("mesures").split(",")]
        mesures[0] = int(mesures[0])
        mesures[1] = int(mesures[1])
        mesures[2] = int(mesures[2])
        mesures[3] = int(mesures[3])
        mesures.append(0)
    except Exception as e:
        log_msg = "FAILED mesures format is incorrect - {}".format(e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "WARNING"
        )
        plogger.log(log_msg, severity="WARNING", logger_name=LOGGER_NAME_VAL)

    json_file = "/input/sizing-system/setting/params_pred.json"
    if not os.path.exists(json_file):
        log_msg = "FAILED json file does not exist {}".format(json_file)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        json_ = json.load(open(json_file, "r"))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
        log_msg = "FAILED to load {} - {} - {}".format(json_file, e, trace_back)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    if mesures is not None:
        json_["mesures_morpho"] = mesures
        json_["batch"] = 0
    else:
        json_["batch"] = 1
        try:
            json_["input_df"] = request.args.get("input_df")
            json_["input_df_w_pred"] = json_["input_df"]
        except Exception as e:
            log_msg = "Bad Request - {}".format(e)
            log_msg = plogger.create_log_message(
                log_msg, __file__, gen_morpho.__name__, "ERROR"
            )
            plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
            out = {"success": False, "exception": log_msg}
            res = json.dumps(out, indent=4)
            resp = Response(res)
            resp.headers["Content-Type"] = "application/json"
            return resp, 400

    SESSION_ID = gen_run_id(request.args.get("SESSION_ID"))
    try:
        p, p_detail, p_stat = pred_morphotype.pred_morpho(json_, SESSION_ID)
    except Exception as e:
        log_msg = "FAILED to obtain prediction - {}".format(e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    out = {
        "success": True,
        "result": p,
        "statistiques": p_stat,
        "detail": p_detail,
        "run_id": SESSION_ID,
    }
    res = json.dumps(out, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/pred_taillant/pred", methods=["POST"])
def pred():
    log_msg = ">> POST /pred_taillant/pred"
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    mesures = None
    try:
        mesures = [c for c in request.args.get("mesures").split(",")]
        mesures[0] = int(mesures[0])
        mesures[1] = int(mesures[1])
        mesures[2] = int(mesures[2])
        mesures[3] = int(mesures[3])
        mesures[5] = int(mesures[5])
        mesures[6] = int(mesures[6])
    except Exception as e:
        log_msg = "FAILED mesures format is incorrect - {}".format(e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, pred.__name__, "WARNING"
        )
        plogger.log(log_msg, severity="WARNING", logger_name=LOGGER_NAME_VAL)
        mesures = None

    json_file = "/input/sizing-system/setting/params_pred.json"

    if not os.path.exists(json_file):
        log_msg = "FAILED json file does not exist {}".format(json_file)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        json_ = json.load(open(json_file, "r"))
    except Exception as e:
        log_msg = "FAILED to load {} - {}".format(json_file, e)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    if mesures is not None:
        json_["mesures"] = mesures
        json_["batch"] = 0
    else:
        json_["batch"] = 1
        try:
            json_["input_df"] = request.args.get("input_df")
        except Exception as e:
            log_msg = "Bad Request - {}".format(e)
            log_msg = plogger.create_log_message(
                log_msg, __file__, gen_morpho.__name__, "ERROR"
            )
            plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
            out = {"success": False, "exception": log_msg}
            res = json.dumps(out, indent=4)
            resp = Response(res)
            resp.headers["Content-Type"] = "application/json"
            return resp, 400

    SESSION_ID = gen_run_id(
        request.args.get("SESSION_ID")
    )  

    try:
        p = pred_mensuration.pred(json_, SESSION_ID)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
        log_msg = "FAILED to obtain prediction - {} - {}".format(e, trace_back)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    out = {"success": True, "result": p}
    res = json.dumps(out, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    return resp


def split_param(param_name, param_val):
    if param_val is None:
        return None
    try:
        return [float(c) for c in param_val.split(",")]
    except Exception as e:
        log_msg = "ERROR:  {} format is incorrect {} - {}".format(
            param_name, param_val, e
        )
        log_msg = plogger.create_log_message(
            log_msg, __file__, split_param.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise


@app.route("/pred_taillant/mesures_et_morpho", methods=["POST"])
def pred_mesure_et_gen_morpho():
    print("PRINTING REQ FILES")
    print(request.files)
    log_msg = ">> POST /UPLOAD"
    log_msg = plogger.create_log_message(
        log_msg, __file__, pred_mesure_et_gen_morpho.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    SESSION_ID = gen_run_id(request.args.get("SESSION_ID"))
    upload_resp, resp_code = upload_file_helper(request, SESSION_ID)
    if resp_code != 200:
        return upload_resp, resp_code
    else:
        file = request.files["file"]
        file_name = SESSION_ID + "_" + secure_filename(file.filename)
    log_msg = ">> POST /pred_taillant/pred"
    log_msg = plogger.create_log_message(
        log_msg, __file__, pred_mesure_et_gen_morpho.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    mesures = None

    json_file = "/input/sizing-system/setting/params_pred.json"
    if not os.path.exists(json_file):
        log_msg = "FAILED json file does not exist {}".format(json_file)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        json_ = json.load(open(json_file, "r"))
    except Exception as e:
        log_msg = "FAILED to load {} - {}".format(json_file, e)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    json_["batch"] = 1
    try:
        json_["input_df"] = file_name

    except Exception as e:
        log_msg = "Bad Request - {}".format(e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        p = pred_mensuration.pred(json_, SESSION_ID)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
        log_msg = "FAILED to obtain prediction - {} - {}".format(e, trace_back)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    # os.system("gsutil cp "+bucket+"output/sizing-system/result/pred_mens/"+p+" " + bucket + "input/sizing-system/data/COPIED_"+SESSION_ID+"_predicted_atts.csv")
    # upload_blob(storage_client, BUCKET_NAME,"output/sizing-system/result/pred_mens/"+p  ,
    # 	"input/sizing-system/data/COPIED_"+SESSION_ID+"_predicted_atts.csv" )
    # copy_blob(storage_client, BUCKET_NAME, "output/sizing-system/result/pred_mens/"+p ,  "input/sizing-system/data/COPIED_"+SESSION_ID+"_predicted_atts.csv",LOGGER_NAME_VAL )

    ###########################################		MORPHO		##########################

    log_msg = ">> POST /pred_taillant/morpho"
    log_msg = plogger.create_log_message(log_msg, __file__, gen_morpho.__name__, "INFO")
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    mesures = None
    json_file = "/input/sizing-system/setting/params_pred.json"
    if not os.path.exists(json_file):
        log_msg = "FAILED json file does not exist {}".format(json_file)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        json_ = json.load(open(json_file, "r"))
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
    json_["input_df_w_pred"] = SESSION_ID + "_predicted_atts.csv"
    json_["batch"] = True

    try:
        p2, p2_detail, p2_stat = pred_morphotype.pred_morpho(json_, SESSION_ID)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
        log_msg = "FAILED to obtain prediction - {}".format(trace_back)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_morpho.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    out = {
        "success": True,
        "result": p2,
        "statistiques": p2_stat,
        "detail": p2_detail,
        "run_id": SESSION_ID,
    }
    res = json.dumps(out, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/taillant/poll", methods=["POST"])
def sessions():
    try:
        type_alg = request.args.get("methode")
        session_id = request.args.get("SESSION_ID")
        env_path = pref + "output/sizing-system/" + session_id + "/env.txt"
        # os.system(
        #    "gsutil cp " + bucket + "output/sizing-system/result/" + type_alg + "/" + session_id + "/env.txt " + pref + "output/sizing-system/" + session_id + "/")
        dd = download_blob(
            storage_client,
            BUCKET_NAME,
            "output/sizing-system/result/" + type_alg + "/" + session_id + "/env.txt",
            env_path,
        )
        with open(env_path, "r") as fp:
            env_ = fp.readline()
        if env_.strip() == "Ok":
            env_path = pref + "output/sizing-system/" + session_id + "/res_gen.zip"
            dd = download_blob(
                storage_client,
                BUCKET_NAME,
                "output/sizing-system/result/"
                + type_alg
                + "/"
                + session_id
                + "/res_gen.zip",
                env_path,
            )  # res=pd.read_csv(pref+"output/sizing-system/" + session_id + "/e")

        out = {"success": True, "result": env_}
    except Exception as e:
        error_msg = "FAILED session not found - {}".format(e)
        out = {"success": False, "exception": error_msg}
    res = json.dumps(out, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/taillant/get_results_json", methods=["POST"])
def send_json():
    try:
        type_alg = request.args.get("methode").strip()
        session_id = request.args.get("SESSION_ID").strip()
        main_output_dir = (
            "gs://smart-fashion-data/output/sizing-system/result/"
            + type_alg
            + "/"
            + session_id
            + "/"
        )
        list_dirs_all = list_dirs_from_gs(storage_client, main_output_dir)
        list_dirs_keep = {}
        download_path = "../" + session_id + ".zip"
        dd = download_blob(
            storage_client,
            BUCKET_NAME,
            "output/sizing-system/result/" + type_alg + "/zip/" + session_id + ".zip",
            download_path,
        )
        with zipfile.ZipFile("../" + session_id + ".zip", "r") as zip_ref:
            zip_ref.extractall("../" + session_id + "-extract")
        for l in list_dirs_all:
            if "affectation" in l:
                list_dirs_keep[secure_filename(l)] = pd.read_csv(
                    "../" + session_id + "-extract/" + l, names=["size number"]
                ).to_json()
            elif "result" in l:
                list_dirs_keep[secure_filename(l)] = pd.read_csv(
                    "../" + session_id + "-extract/" + l
                ).to_json()
            elif "centroide" in l:
                list_dirs_keep[secure_filename(l)] = pd.read_csv(
                    "../" + session_id + "-extract/" + l
                )[
                    pd.read_csv("../" + session_id + "-extract/" + l).columns[1:]
                ].to_json()
        os.system("rm -r ../" + session_id + "-extract ")
        os.system("rm  ../" + session_id + ".zip ")
        out = {"success": "True", "result": list_dirs_keep}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 200
    except Exception as e:
        # couldn't find file
        exc_type, exc_value, exc_tb = sys.exc_info()
        trace_back = traceback.format_exception(exc_type, exc_value, exc_tb)
        log_msg = "ERROR: couldn't find result file (did you provide a correct SESSION_ID and methode) - {}".format(
            trace_back
        )
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400


@app.route("/taillant/get_results", methods=["POST"])
def send_zip():
    try:
        type_alg = request.args.get("methode")
        session_id = request.args.get("SESSION_ID")
        return send_file(
            "/output/sizing-system/" + session_id + "/res_gen.zip", as_attachment=True
        )
    except Exception as e:
        # couldn't find file
        log_msg = "ERROR: couldn't find result file (did you provide a correct SESSION_ID and methode) - {}".format(
            e
        )
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400


@app.route("/taillant/quadrillage", methods=["POST"])
def gen_taillant_quad():
    log_msg = ">> POST /taillant/quadrillage"
    log_msg = plogger.create_log_message(
        log_msg, __file__, gen_taillant_quad.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    BMI = request.args.get("BMI")
    try:
        BMI = split_param("BMI", BMI)

    except Exception as e:
        # "BMI format should be 'min,max' or None"
        log_msg = "ERROR: format is incorrect - {}".format(e)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
        # BMI=np.array(BMI)

    Age = request.args.get("Age")
    try:
        Age = split_param("Age", Age)

    except Exception as e:
        log_msg = "ERROR: format is incorrect - {}".format(e)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    if not os.path.exists(user_settings_GBSS):
        log_msg = "FAILED file does not exist - {}".format(user_settings_GBSS)
        log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400

    try:
        admin_params = json.load(open(user_settings_GBSS, "r"))
    except Exception as e:
        log_msg = "FAILED to load {} - {}".format(user_settings_GBSS, e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_taillant_quad.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp

    json_ = {}
    SESSION_ID = gen_run_id(
        request.args.get("SESSION_ID")
    )  # request.args.get('SESSION_ID') + "-" + str(round(time.monotonic() * 1000))
    os.mkdir(pref + "/output/sizing-system/" + SESSION_ID)
    env_["session_" + SESSION_ID] = 0.0
    json_["columns"] = [c for c in request.args.get("columns").split(",")]
    json_["input_df"] = request.args.get("input_df")
    json_["minclasses"] = 0  # int(request.args.get('minClasses'))
    json_["maxclasses"] = int(request.args.get("maxClasses"))
    json_["Output_file"] = "output/sizing-system/result/quadrillage/" + SESSION_ID + ""
    json_["alpha_"] = admin_params["alpha_"]
    json_["beta_"] = admin_params["beta_"]

    try:
        create_app(json_, SESSION_ID, env_, BMI, Age, True)
    except Exception as e:
        log_msg = "FAILED to launch quadrillage - {}".format(e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_taillant_quad.__name__, "ERROR"
        )
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp

    log_msg = "STARTED quadrillage {}".format(SESSION_ID)
    log_msg = plogger.create_log_message(
        log_msg, __file__, gen_taillant.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
    out = {"success": True, "result": SESSION_ID}
    res = json.dumps(out, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    return resp


@app.route("/taillant/genetique", methods=["POST"])
def gen_taillant():
    log_msg = ">> POST /taillant/genetique"
    log_msg = plogger.create_log_message(
        log_msg, __file__, gen_taillant.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    BMI = request.args.get("BMI")
    try:
        BMI = split_param("BMI", BMI)

    except Exception as e:
        # "BMI format should be 'min,max' or None"
        log_msg = "ERROR: format is incorrect - {}".format(e)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
    Age = request.args.get("Age")
    try:
        Age = split_param("Age", Age)
    except Exception as e:
        log_msg = "ERROR: format is incorrect - {}".format(e)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp, 400
    try:
        admin_params = json.load(open(user_settings, "r"))
    except Exception as e:
        log_msg = "FAILED to load {} - {}".format(user_settings, e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_taillant.__name__, "ERROR"
        )
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp

    json_ = {}
    SESSION_ID = gen_run_id(
        request.args.get("SESSION_ID")
    )  # request.args.get('SESSION_ID') + "-" + str(round(time.monotonic() * 1000))
    os.mkdir(pref + "/output/sizing-system/" + SESSION_ID)
    env_["session_" + SESSION_ID] = 0
    # json_=json.loads(json_)
    json_["SESSION_ID"] = SESSION_ID
    json_["columns"] = [c for c in request.args.get("columns").split(",")]
    json_["input_df"] = request.args.get("input_df")
    json_["category"] = request.args.get("category")
    json_["SOM"] = admin_params["SOM"]
    json_["BBSS"] = int(request.args.get("BBSS"))
    json_["numGen"] = admin_params["numGen"]
    json_["halfPopSize"] = admin_params["halfPopSize"]
    json_["K"] = admin_params["K"]
    json_["seed"] = admin_params["seed"]
    json_["minClasses"] = int(request.args.get("minClasses"))
    json_["maxClasses"] = int(request.args.get("maxClasses"))
    json_["Output_file"] = "output/sizing-system/result/genetique/" + SESSION_ID + ""
    print(json_)
    # json_["morpho_file"]=request.args.get('morpho_file')

    try:
        create_app(json_, SESSION_ID, env_, BMI, Age, False)
    except Exception as e:
        log_msg = "FAILED to launch quadrillage - {}".format(e)
        log_msg = plogger.create_log_message(
            log_msg, __file__, gen_taillant.__name__, "ERROR"
        )
        out = {"success": False, "exception": log_msg}
        res = json.dumps(out, indent=4)
        resp = Response(res)
        resp.headers["Content-Type"] = "application/json"
        return resp

    log_msg = "STARTED genetic algorithm {}".format(SESSION_ID)
    log_msg = plogger.create_log_message(
        log_msg, __file__, gen_taillant.__name__, "INFO"
    )
    plogger.log(log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)
    out = {"success": True, "result": SESSION_ID}
    res = json.dumps(out, indent=4)
    resp = Response(res)
    resp.headers["Content-Type"] = "application/json"
    return resp


if __name__ == "__main__":
    try:
        main_log = "Listening on {}".format(os.getenv("PORT", PORT))
        main_log = plogger.create_log_message(main_log, __file__, "__main__", "INFO")
        plogger.log(main_log, severity="INFO", logger_name=LOGGER_NAME_VAL)
        http_server = WSGIServer(("0.0.0.0", int(os.getenv("PORT", PORT))), app)
        http_server.serve_forever()
    except Exception as err:
        main_log = "FAILED to start  WSGIServer - {}".format(err)
        main_log = plogger.create_log_message(
            main_log, __file__, gen_taillant.__name__, "ERROR"
        )
        plogger.log(main_log, severity="ERROR", logger_name=LOGGER_NAME_VAL)
