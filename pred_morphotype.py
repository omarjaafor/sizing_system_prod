# #Permet de prédire les morphotypes pour des clientes, sert à filtrer par morphotype ou générer des stats.

import copy
import json
import os

import numpy as np
import pandas as pd

from lib import plogger
from lib.constants import *
from lib.gcp_function import (
    get_list_local_files,
    upload_blobs,
    download_blob,
    upload_blob,
    download_blobs,
)
from google.cloud import storage

storage_client = storage.Client()


def pred_morpho(json_, run_id):
    """
    Predicts the morphological type of female users

    Parameters
    ----------
    json_: type,
        batch: boolean, indicates whether to run the function in batch mode or one shot mode
        mesures_morpho: array<float>, contains stature, hip, bust, waist if batch =True, None if batch=False
    Returns
    -------
    message:json, contains the morpholgical distribution of female users if batch=True or the morphological class of one female user if batch=False
    example
    Case batch=True:
        {
        'A' : 0.33,
        'H':0.16,
        'V':0,
        'X':0.25,
        'O':0.25
        }
    Case batch=False
        {

        'A':0,
        'H':1,
        'V':0,
        'X':0,
        'O':0
        }

    """
    log_params = {"json_": json_}
    log_msg = "START pred_morpho({params})".format(params=log_params)
    log_msg = plogger.create_log_message(
        log_msg, __file__, pred_morpho.__name__, "INFO"
    )
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    # SI BATCH LECTURE DU XLS
    if json_["batch"]:
        df = pd.read_csv("/input/sizing-system/data/" + json_["input_df_w_pred"])
        pd.set_option("display.max_columns", 500)
    # SI ONE SHOT TRANSFORMATION DU json EN XLS
    else:
        df = pd.DataFrame.from_records([json_["mesures_morpho"]])
        df.columns = ["stature", "hip", "bust", "waist", "under_bust"]

    # CREATION DES REGLES D AFFECTATION
    tt_tp_dm_min = -120
    tt_tp_dm_max = 149

    tt_tp_dp_min = -358
    tt_tp_dp_max = -160

    tt_tp_eq_min = -160
    tt_tp_eq_max = -120

    tt_tb_dm_min = -180
    tt_tb_dm_max = 76

    tt_tb_dp_min = -439
    tt_tb_dp_max = -235

    tt_tb_eq_min = -235
    tt_tb_eq_max = -180

    set_tt_tp_dm = set(np.where(df["waist"] - df["bust"] >= tt_tp_dm_min)[0])
    set_tt_tp_dp = set(np.where(df["waist"] - df["bust"] <= tt_tp_dp_max)[0])
    set_tt_tp_eq = set(np.where(df["waist"] - df["bust"] >= tt_tp_eq_min)[0]) & set(
        np.where(df["waist"] - df["bust"] <= tt_tp_eq_max)[0]
    )

    set_tt_tb_dm = set(np.where(df["waist"] - df["hip"] >= tt_tb_dm_min)[0])
    set_tt_tb_dp = set(np.where(df["waist"] - df["hip"] <= tt_tb_dp_max)[0])
    set_tt_tb_eq = set(np.where(df["waist"] - df["hip"] >= tt_tb_eq_min)[0]) & set(
        np.where(df["waist"] - df["hip"] <= tt_tb_eq_max)[0]
    )

    S = {}
    S["O"] = set_tt_tp_dm & set_tt_tb_dm
    S["Trapeze"] = set_tt_tp_dm & set_tt_tb_eq
    S["A"] = set_tt_tp_dm & set_tt_tb_dp
    S["None 1"] = set_tt_tp_eq & set_tt_tb_dm
    S["H"] = set_tt_tp_eq & set_tt_tb_eq
    S["None 2"] = set_tt_tp_eq & set_tt_tb_dp
    S["V"] = set_tt_tp_dp & set_tt_tb_dm
    S["None 3"] = set_tt_tp_dp & set_tt_tb_eq
    S["X"] = set_tt_tp_dp & set_tt_tb_dp
    S["O"] = S["O"] | S["None 1"]
    S["A"] = S["A"] | S["None 2"]
    S["V"] = S["V"] | S["None 3"]
    del S["None 1"]
    del S["None 2"]
    del S["None 3"]
    S["O"] = np.array(list(S["O"]))
    S["Trapeze"] = np.array(list(S["Trapeze"]))
    S["A"] = np.array(list(S["A"]))
    S["H"] = np.array(list(S["H"]))
    S["V"] = np.array(list(S["V"]))
    S["X"] = np.array(list(S["X"]))

    # AFFECTATION AUX MORPHOS TYPES
    P = {}
    P["O"] = len(set_tt_tp_dm & set_tt_tb_dm) / df.shape[0]
    P["Trapeze"] = len(set_tt_tp_dm & set_tt_tb_eq) / df.shape[0]
    P["A"] = len(set_tt_tp_dm & set_tt_tb_dp) / df.shape[0]
    P["None 1"] = len(set_tt_tp_eq & set_tt_tb_dm) / df.shape[0]
    P["H"] = len(set_tt_tp_eq & set_tt_tb_eq) / df.shape[0]
    P["None 2"] = len(set_tt_tp_eq & set_tt_tb_dp) / df.shape[0]
    P["V"] = len(set_tt_tp_dp & set_tt_tb_dm) / df.shape[0]
    P["None 3"] = len(set_tt_tp_dp & set_tt_tb_eq) / df.shape[0]
    P["X"] = len(set_tt_tp_dp & set_tt_tb_dp) / df.shape[0]

    # AFFINAGE DU RESULTAT
    M = copy.deepcopy(P)
    M["O"] = M["O"] + M["None 1"]
    del M["None 1"]
    M["A"] = M["A"] + M["None 2"]
    del M["None 2"]
    M["V"] = M["V"] + M["None 3"]
    del M["None 3"]

    output_file = (
        "/output/sizing-system/result/morpho/batch/" + run_id + "_result_morpho.json"
    )
    with open(output_file, "w") as fp:
        json.dump(M, fp)

    # os.system("gsutil cp /input/sizing-system/data/morpho.json gs://smart-fashion-data/output/sizing_system/result/" + json_["morpho_file"])
    stats_ = {}
    if json_["batch"]:
        result = json_["morpho_file"]
        upload_blob(
            storage_client,
            BUCKET_NAME,
            output_file,
            "output/sizing-system/result/morpho/batch/"
            + run_id
            + "_result_morpho.json",
        )
        df["morpho"] = "ND"
        df.loc[S["O"], "morpho"] = "O"
        stats_["O"] = {}
        try:
            stats_["O"]["stature"] = [
                df.loc[S["O"], "stature"].mean(),
                df.loc[S["O"], "stature"].min(),
                df.loc[S["O"], "stature"].max(),
            ]
            stats_["O"]["hip"] = [
                df.loc[S["O"], "hip"].mean(),
                df.loc[S["O"], "hip"].min(),
                df.loc[S["O"], "hip"].max(),
            ]
            stats_["O"]["bust"] = [
                df.loc[S["O"], "bust"].mean(),
                df.loc[S["O"], "bust"].min(),
                df.loc[S["O"], "bust"].max(),
            ]
            stats_["O"]["under_bust"] = [
                df.loc[S["O"], "under_bust"].mean(),
                df.loc[S["O"], "under_bust"].min(),
                df.loc[S["O"], "under_bust"].max(),
            ]
        except:
            None
        df.loc[S["Trapeze"], "morpho"] = "Trapeze"
        stats_["Trapeze"] = {}
        try:
            stats_["Trapeze"]["stature"] = [
                df.loc[S["Trapeze"], "stature"].mean(),
                df.loc[S["Trapeze"], "stature"].min(),
                df.loc[S["Trapeze"], "stature"].max(),
            ]
            stats_["Trapeze"]["hip"] = [
                df.loc[S["Trapeze"], "hip"].mean(),
                df.loc[S["Trapeze"], "hip"].min(),
                df.loc[S["Trapeze"], "hip"].max(),
            ]
            stats_["Trapeze"]["bust"] = [
                df.loc[S["Trapeze"], "bust"].mean(),
                df.loc[S["Trapeze"], "bust"].min(),
                df.loc[S["Trapeze"], "bust"].max(),
            ]
            stats_["Trapeze"]["under_bust"] = [
                df.loc[S["Trapeze"], "under_bust"].mean(),
                df.loc[S["Trapeze"], "under_bust"].min(),
                df.loc[S["Trapeze"], "under_bust"].max(),
            ]
        except:
            None
        df.loc[S["A"], "morpho"] = "A"
        stats_["A"] = {}
        try:
            stats_["A"]["stature"] = [
                df.loc[S["A"], "stature"].mean(),
                df.loc[S["A"], "stature"].min(),
                df.loc[S["A"], "stature"].max(),
            ]
            stats_["A"]["hip"] = [
                df.loc[S["A"], "hip"].mean(),
                df.loc[S["A"], "hip"].min(),
                df.loc[S["A"], "hip"].max(),
            ]
            stats_["A"]["bust"] = [
                df.loc[S["A"], "bust"].mean(),
                df.loc[S["A"], "bust"].min(),
                df.loc[S["A"], "bust"].max(),
            ]
            stats_["A"]["under_bust"] = [
                df.loc[S["A"], "under_bust"].mean(),
                df.loc[S["A"], "under_bust"].min(),
                df.loc[S["A"], "under_bust"].max(),
            ]
        except:
            None

        stats_["H"] = {}
        df.loc[S["H"], "morpho"] = "H"
        try:
            stats_["H"]["stature"] = [
                df.loc[S["H"], "stature"].mean(),
                df.loc[S["H"], "stature"].min(),
                df.loc[S["H"], "stature"].max(),
            ]
            stats_["H"]["hip"] = [
                df.loc[S["H"], "hip"].mean(),
                df.loc[S["H"], "hip"].min(),
                df.loc[S["H"], "hip"].max(),
            ]
            stats_["H"]["bust"] = [
                df.loc[S["H"], "bust"].mean(),
                df.loc[S["H"], "bust"].min(),
                df.loc[S["H"], "bust"].max(),
            ]
            stats_["H"]["under_bust"] = [
                df.loc[S["H"], "under_bust"].mean(),
                df.loc[S["H"], "under_bust"].min(),
                df.loc[S["H"], "under_bust"].max(),
            ]
        except:
            None

        stats_["V"] = {}
        df.loc[S["V"], "morpho"] = "V"
        try:
            stats_["V"]["stature"] = [
                df.loc[S["V"], "stature"].mean(),
                df.loc[S["V"], "stature"].min(),
                df.loc[S["V"], "stature"].max(),
            ]
            stats_["V"]["hip"] = [
                df.loc[S["V"], "hip"].mean(),
                df.loc[S["V"], "hip"].min(),
                df.loc[S["V"], "hip"].max(),
            ]
            stats_["V"]["bust"] = [
                df.loc[S["V"], "bust"].mean(),
                df.loc[S["V"], "bust"].min(),
                df.loc[S["V"], "bust"].max(),
            ]
            stats_["V"]["under_bust"] = [
                df.loc[S["V"], "under_bust"].mean(),
                df.loc[S["V"], "under_bust"].min(),
                df.loc[S["V"], "under_bust"].max(),
            ]
        except:
            None

        stats_["X"] = {}
        df.loc[S["X"], "morpho"] = "X"
        try:
            stats_["X"]["stature"] = [
                df.loc[S["X"], "stature"].mean(),
                df.loc[S["X"], "stature"].min(),
                df.loc[S["X"], "stature"].max(),
            ]
            stats_["X"]["hip"] = [
                df.loc[S["X"], "hip"].mean(),
                df.loc[S["X"], "hip"].min(),
                df.loc[S["X"], "hip"].max(),
            ]
            stats_["X"]["bust"] = [
                df.loc[S["X"], "bust"].mean(),
                df.loc[S["X"], "bust"].min(),
                df.loc[S["X"], "bust"].max(),
            ]
            stats_["X"]["under_bust"] = [
                df.loc[S["X"], "under_bust"].mean(),
                df.loc[S["X"], "under_bust"].min(),
                df.loc[S["X"], "under_bust"].max(),
            ]
        except:
            None
        stats_["ALL"] = {}
        stats_["ALL"]["stature"] = [
            df["stature"].mean(),
            df["stature"].min(),
            df["stature"].max(),
        ]
        stats_["ALL"]["hip"] = [df["hip"].mean(), df["hip"].min(), df["hip"].max()]
        stats_["ALL"]["bust"] = [df["bust"].mean(), df["bust"].min(), df["bust"].max()]
        stats_["ALL"]["under_bust"] = [
            df["under_bust"].mean(),
            df["under_bust"].min(),
            df["under_bust"].max(),
        ]

        output_file = (
            "/output/sizing-system/result/morpho/batch/"
            + run_id
            + "_detail_mensurations_morpho.csv"
        )
        df.to_csv(output_file)

        upload_blob(
            storage_client,
            BUCKET_NAME,
            output_file,
            "output/sizing-system/result/morpho/batch/"
            + run_id
            + "_detail_mensurations_morpho.csv",
        )

        output_file = (
            "/output/sizing-system/result/morpho/batch/" + run_id + "_statistics.json"
        )

        with open(output_file, "w") as fp:
            json.dump(stats_, fp)

        upload_blob(
            storage_client,
            BUCKET_NAME,
            output_file,
            "output/sizing-system/result/morpho/batch/" + run_id + "_statistics.json",
        )

        return (
            run_id + "_result_morpho.json",
            run_id + "_detail_mensurations_morpho.csv",
            run_id + "_statistics.json",
        )
    else:
        result = M

    log_msg = "END pred_morpho({params}) -> {returns}".format(
        params=log_params, returns=result
    )
    log_msg = plogger.create_log_message(
        log_msg, __file__, pred_morpho.__name__, "INFO"
    )
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    return result, None, None





