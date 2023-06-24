import os

import joblib
import numpy as np
import pandas as pd
import xlsx2csv as xls
import codecs
from lib.constants import *
from lib import plogger
import json
from shutil import copy
from lib.gcp_function import (
    get_list_local_files,
    upload_blobs,
    download_blob,
    upload_blob,
    download_blobs,
)
from google.cloud import storage

storage_client = storage.Client()


# CONVERT EXCEL TO CSV
def fast_read_excel(f, sheet_id=1):
    """
    Read a excel file (given a path and a sheet id) and load a `pd.DataFrame`
    """
    try:
        csv_filename = f.replace(".xlsx", "") + "_" + str(sheet_id) + ".csv"
        xls.Xlsx2csv(f, outputencoding="utf-8").convert(csv_filename, sheetid=sheet_id)
        df = pd.read_csv(csv_filename, sep=",", low_memory=False)
    except Exception as e:
        message = "FAILED reading excel file {} - {}".format(f, e)
        message = plogger.create_log_message(
            message, __file__, fast_read_excel.__name__, "ERROR"
        )
        plogger.log(message=message, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise
    else:
        message = "SUCCEED loading excel file ({} - sheet_id={}) into Dataframe".format(
            f, sheet_id
        )
        message = plogger.create_log_message(
            message, __file__, fast_read_excel.__name__, "INFO"
        )
        plogger.log(message=message, severity="INFO", logger_name=LOGGER_NAME_VAL)

    return df


def load_joblib(path):
    """
    read a joblib file from `path` and load an object
    """
    try:
        obj = joblib.load(path)
    except Exception as e:
        message = "FAILED loading joblib at {} - {}".format(path, e)
        message = plogger.create_log_message(
            message, __file__, load_joblib.__name__, "ERROR"
        )
        plogger.log(message=message, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise
    else:
        message = "SUCCEED loading joblib file {}".format(path)
        message = plogger.create_log_message(
            message, __file__, load_joblib.__name__, "INFO"
        )
        plogger.log(message=message, severity="INFO", logger_name=LOGGER_NAME_VAL)

    return obj


def load_np(path):
    """
    Read a numpy file
    """
    try:
        obj = np.load(path, allow_pickle=True)
    except Exception as e:
        message = "FAILED loading numpy file at {} - {}".format(path, e)
        message = plogger.create_log_message(
            message, __file__, load_joblib.__name__, "ERROR"
        )
        plogger.log(message=message, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise
    else:
        message = "SUCCEED loading numpy file {}".format(path)
        message = plogger.create_log_message(
            message, __file__, load_np.__name__, "INFO"
        )
        plogger.log(message=message, severity="INFO", logger_name=LOGGER_NAME_VAL)

    return obj


# "/home/ext-ojaafor/sizing-system/data/sondage_cleaned.csv"


# PREDICT bust,hip,waist, underbust
def pred(json_, run_id, REG_RATE=1 / 3):
    """
    predicts missing morphological attributs of female clients (bust,hip,waist,under bust)

    Parameters
    ----------
    json_: json,
        batch: boolean, indicates whether to execute the function in batch mode or one shot mode
        mesures: available morphological attributes used to predict missing morphological attributes (stature(mm),weight(kg),age(years),tour sg(mm),bonnet(A,B,C,D,E),tour hanche(mm))
    run_id: string,
        obtained from flask_app, timestamp-sess_id
    REG_RATE: float,
        desc parametre metier
    Returns
    -------
    Case batch=True
        returnedVal:json,predicted values(hip,bust,waist)
    Case batch=False
        returnedVal:run_id/file_name, contains the predicted values (hip,bust,waist)
    """
    # verifie si c'est one shot ou batch,
    labels = {"run_id": run_id}
    mode_pred = "batch" if json_["batch"] else "one shot"
    log_msg = "START bust, hip, waist, underbust prediction function in mode {} (run_id = {})".format(
        mode_pred, run_id
    )
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )

    if json_["batch"]:
        df = fast_read_excel("/input/sizing-system/data/" + json_["input_df"])

    else:
        # si c'est one shot, transforme le json en dataframe pour faire les pred
        df = pd.DataFrame.from_records([json_["mesures"]])
        df.columns = [
            "stature",
            "weight",
            "age",
            "tour sg",
            "bonnet",
            "tour taille",
            "tour hanche",
        ]

    # load des données utilisées pour la régularisation (semi supervisée)
    log_msg = (
        "LOADING data used for regularization (semi supervised) (run_id = {})".format(
            run_id
        )
    )
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )

    reg_toursg_bonnet = load_np(
        "/output/sizing-system/models/regularizors/reg_toursg_bonnet.npy"
    )
    reg_tt = load_np("/output/sizing-system/models/regularizors/reg_tt.npy")
    reg_th = load_np("/output/sizing-system/models/regularizors/reg_th.npy")

    # load les models
    log_msg = "LOADING models (run_id = {})".format(run_id)
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )

    mlr = {}
    Folder_models = "/output/sizing-system/models/mlr/"
    for c in [
        "Hip Circumference, Maximum (mm)",
        "Chest Circumference (mm)",
        "Waist Circumference, Pref (mm)",
        "BustChest Circumference Under Bust (mm)",
    ]:
        print("mlr" + c + ".pkl")
        mlr[c] = load_joblib(Folder_models + "mlr" + c + ".pkl")
        # joblib.dump()
    min_max_scaler = {}
    for c in [
        "Hip Circumference, Maximum (mm)",
        "Chest Circumference (mm)",
        "Waist Circumference, Pref (mm)",
        "BustChest Circumference Under Bust (mm)",
    ]:
        print("min_max_scaler" + c + ".pkl")
        min_max_scaler[c] = load_joblib(Folder_models + "min_max_scaler" + c + ".pkl")

    poly = {}
    for c in [
        "Hip Circumference, Maximum (mm)",
        "Chest Circumference (mm)",
        "Waist Circumference, Pref (mm)",
        "BustChest Circumference Under Bust (mm)",
    ]:
        print("poly" + c + ".pkl")
        poly[c] = load_joblib(Folder_models + "poly" + c + ".pkl")

        # prend uniquement les attributs utilisés dans la prediction
        predictor_ = df[["age", "weight", "stature"]]

    # Effectue les predictions
    log_msg = "GETTING predictions (run_id = {})".format(run_id)
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )

    for c in [
        "Hip Circumference, Maximum (mm)",
        "Chest Circumference (mm)",
        "Waist Circumference, Pref (mm)",
        "BustChest Circumference Under Bust (mm)",
    ]:
        predictor_scaled = min_max_scaler[c].transform(predictor_)

        predictor_poly = poly[c].transform(predictor_scaled)
        predicted_ = mlr[c].predict(predictor_poly.astype(float))

        predictor_ = pd.concat(
            [predictor_, pd.DataFrame(predicted_)], axis=1, ignore_index=True
        )

    predictor_["tour sg"] = df["tour sg"]
    predictor_["bonnet"] = df["bonnet"]
    predictor_["tour taille"] = df["tour taille"]
    predictor_["tour hanche"] = df["tour hanche"]
    columns = {
        0: "Age (Years)",
        1: "Weight (kg)",
        2: "Stature (mm)",
        3: "Hip Circumference, Maximum (mm)",
        4: "Chest Circumference (mm)",
        5: "Waist Circumference, Pref (mm)",
        6: "BustChest Circumference Under Bust (mm)",
    }
    predictor_.rename(columns=columns, inplace=True)

    # Regularisation bust
    log_msg = "REGULARIZATION bust (run_id = {})".format(run_id)
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )

    tsg = np.unique(predictor_["tour sg"])
    bonnet = np.unique(predictor_["bonnet"])

    for t in tsg:
        s1 = set(np.where(predictor_["tour sg"].values == t)[0])

        for b in bonnet:
            if str(t) + "_" + b in reg_toursg_bonnet:
                s2 = set(np.where(predictor_["bonnet"].values == b)[0])
                su = np.array(list(s1 & s2))
                predictor_.loc[su, "Chest Circumference (mm)"] = (
                    1 - REG_RATE
                ) * predictor_.loc[
                    su, "Chest Circumference (mm)"
                ] + REG_RATE * reg_toursg_bonnet[
                    str(t) + "_" + b
                ]

    # regularisation tour de taille
    log_msg = "REGULARIZATION tour de taille (run_id = {})".format(run_id)
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )
    tt = np.unique(predictor_["tour taille"])

    for t in tt:
        if str(t) in reg_tt:
            s1 = set(np.where(predictor_["tour taille"].values == t)[0])
            predictor_.loc[s1, "Waist Circumference, Pref (mm)"] = (
                1 - REG_RATE
            ) * predictor_.loc[
                s1, "Waist Circumference, Pref (mm)"
            ] + REG_RATE * reg_tt[
                str(t)
            ]

    log_msg = "REGULARIZATION tour de hanches (run_id = {})".format(run_id)
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )
    th = np.unique(predictor_["tour hanche"])
    # regularisation tour de hanches
    for t in th:
        if str(t) in reg_th:
            s1 = set(np.where(predictor_["tour taille"].values == t)[0])
            predictor_.loc[s1, "Hip Circumference, Maximum (mm)"] = (
                1 - REG_RATE
            ) * predictor_.loc[
                s1, "Hip Circumference, Maximum (mm)"
            ] + REG_RATE * reg_tt[
                str(t)
            ]

            # transforme les resultats en cm et renomme les colonnes
    predictor_[
        [
            "Stature (mm)",
            "Hip Circumference, Maximum (mm)",
            "Chest Circumference (mm)",
            "Waist Circumference, Pref (mm)",
            "BustChest Circumference Under Bust (mm)",
        ]
    ] /= 10
    predictor_ = predictor_[
        [
            "Stature (mm)",
            "Hip Circumference, Maximum (mm)",
            "Chest Circumference (mm)",
            "Waist Circumference, Pref (mm)",
            "BustChest Circumference Under Bust (mm)",
        ]
    ]

    log_msg = "as path to saved csv file" if json_["batch"] else "as json object"
    log_msg = "RETURNING prediction results {} (run_id = {})".format(log_msg, run_id)
    log_msg = plogger.create_log_message(log_msg, __file__, pred.__name__, "INFO")
    plogger.log(
        message=log_msg, severity="INFO", labels=labels, logger_name=LOGGER_NAME_VAL
    )
    predictor_ *= 10  # transformer en mmm
    # si batche enregistre le csv
    if json_["batch"]:
        predictor_.rename(
            columns={
                "Stature (mm)": "stature",
                "Hip Circumference, Maximum (mm)": "hip",
                "Chest Circumference (mm)": "bust",
                "Waist Circumference, Pref (mm)": "waist",
                "BustChest Circumference Under Bust (mm)": "under_bust",
            },
            inplace=True,
        )
        output_file = (
            "/output/sizing-system/result/pred_mens/batch/"
            + run_id
            + "_predicted_atts.csv"
        )
        predictor_.to_csv(output_file)
        # os.system(
        #    "gsutil cp ../predicted_" + run_id + ".csv " + BUCKET_NAME + "output/sizing-system/result/pred_mens/" + run_id + "/" +
        #    json_["Output_file"])
        copy(output_file, "/input/sizing-system/data/" + run_id + "_predicted_atts.csv")
        upload_blob(
            storage_client,
            BUCKET_NAME,
            output_file,
            "output/sizing-system/result/pred_mens/batch/"
            + run_id
            + "_predicted_atts.csv",
        )
        returnedVal = run_id + "_predicted_atts.csv"
        return returnedVal
    else:
        # si one shot retourne les res
        # todo remove the next lines with json_lst, it is not used and can be removed
        json_lst = []
        for c in predictor_.columns:
            json_lst.append(c + ":" + str(predictor_.loc[0, c]))
        json_str = ",".join(json_lst)
        json_str = "{" + json_str + "}"
        codecs.open("../out.json", "w", "utf-8")
        # returnedVal= {'"hip":' + str(predictor_.iloc[0, 1]) + ',"bust":' + str(predictor_.iloc[0, 2]) + ',"waist":' + str(
        #    predictor_.iloc[0, 3]) +',"under bust":'+str(predictor_.iloc[0,4])+ ''}

        # returnedVal= "{"bust":' + str(predictor_.iloc[0, 2]) + ',"under_bust":' + str(predictor_.iloc[0, 4]) + ',"waist":' + str(
        #    predictor_.iloc[0, 3]) +',"hip":'+str(predictor_.iloc[0,1])+ "}"
        returnedVal = {}
        returnedVal["bust"] = predictor_.iloc[0, 2]
        returnedVal["under_bust"] = predictor_.iloc[0, 4]
        returnedVal["waist"] = predictor_.iloc[0, 3]
        returnedVal["hip"] = predictor_.iloc[0, 1]
        # message_json = json.dumps(returnedVal)
        # enc_message = message_json.encode("utf-8")

        return returnedVal


# json_=json.load(open("params.json","r"))

# pred(json_)
