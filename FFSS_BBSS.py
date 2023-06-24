# AGLO GENETIQUE (ancien gentique.py)
# desc: Ce fichier est utilisé pour gererer un taillant en utilisant l'algorithme genetique. Il orchestre toutes les fonctions de l'algorithme genetique.
# La fonction EXE est appelee par Launcher.py. Elle copy puis lit le input de l'utilisateur en se basant sur run_id.
# La fonction comment par lire les modeles. Ensuite elle appelle SOM.py pour initializer "les solutions retournees". Enfin, elle lance dans
# plusieurs iterations gen_algorithme (une instance de l'algorithme genetique)


# IMPORTS


import codecs
import json
import multiprocessing
import os
from os.path import basename
import zipfile
import joblib
import numpy as np
import pandas as pd

# Load data
import xlsx2csv as xls
from google.cloud import storage
import shutil
import Gen_helper_functions
import SOM
import gen_algorithm
import multi_proc_helper
from lib import plogger
from lib.constants import *
from lib.gcp_function import (
    get_list_local_files,
    upload_blobs,
    download_blob,
    upload_blob,
    download_blobs,
)

# Pour tester en local
# model_path = '/output/models'
# data = '/input/data/'

storage_client = storage.Client()
plogger.create_logger(os.getenv(LOGGER_NAME, LOGGER_NAME_VAL))

# Dossiers crées sur la bucket
pref = "/"
model_path = pref + "output/sizing-system/models"
data = pref + "input/sizing-system/data/"

# PARAMETRES FIXES DE L ALGO GENETIQUE
admin_settings = "params.json"


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), basename(os.path.join(root, file)))


def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move("%s.%s" % (name, format), destination)


def fast_read_excel(f, sheet_id=1):
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


def load_joblib(path, re_raise=True):
    try:
        obj = joblib.load(path)
    except Exception as e:
        severity = ERROR if re_raise else WARNING
        message = "FAILED loading joblib at {} - {}".format(path, e)
        message = plogger.create_log_message(
            message, __file__, load_joblib.__name__, severity
        )
        plogger.log(message=message, severity=severity, logger_name=LOGGER_NAME_VAL)
        if re_raise:
            raise
    else:
        message = "SUCCEED loading joblib file {}".format(path)
        message = plogger.create_log_message(
            message, __file__, load_joblib.__name__, "INFO"
        )
        plogger.log(message=message, severity="INFO", logger_name=LOGGER_NAME_VAL)
    return obj


# json_=json.load(open(admin_settings,"r"))
def EXE(sess_id):
    """
    returns a sizing system using a genetic algorithm explained in : https://ieeexplore.ieee.org/abstract/document/9530202/

    Parameters
    ----------
    json_: json récupéré par la sess_id,
        BMI (list[float]): BMI range
        Age (list(float)): range
        columns (list(str)) : which sizing attributes to use
        BBSS  (bool):  FFSS or BBSS
        genetic alg. params : https://ieeexplore.ieee.org/abstract/document/9530202/
    Returns
    -------
    saves sizing system in : pref+"output/sizing-system/"+sess_id+".zip "+folder_name+"/res_gen.zip
    """
    log_params = {"sess_id": sess_id}
    log_msg = "START genetic algorithm EXE({params})".format(params=log_params)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    # USER INPUT
    user_input_json = "input/sizing-system/setting/input_" + sess_id + ".json"

    log_msg = "DOWNLOAD USER INPUT {}".format(user_input_json)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    download_blob(storage_client, BUCKET_NAME, user_input_json, pref + user_input_json)

    if not os.path.exists(pref + user_input_json):
        log_msg = "FAILED json file does not exist: {}".format(pref + user_input_json)
        log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise FileNotFoundError(
            "json file does not exist: {}".format(pref + user_input_json)
        )

    with open(pref + user_input_json) as json_file:
        json_ = json.load(json_file)

    manager = multiprocessing.Manager()
    env_ = manager.dict()
    # sess_id=json_["sess_id"]
    # VARIABLE CONTENANT LE PROGRES DE L'ALGO
    env_[sess_id] = 0
    BMI = json_["BMI"]
    if BMI is not None:
        BMI = np.array(BMI)

    Age = json_["Age"]
    if Age is not None:
        Age = np.array(Age)

    # fichier temporaire pour enregistrer les resultats (ne pas gerer les exception car migration en cours vers un fichier qui est mirroire de la bucket)
    if not os.path.exists(pref + "output/sizing-system/" + sess_id):
        os.mkdir(pref + "output/sizing-system/" + sess_id)
    if not os.path.exists(pref + "output/sizing-system/" + sess_id + "/results"):
        os.mkdir(pref + "output/sizing-system/" + sess_id + "/results")

    folder_name = pref + "output/sizing-system/" + sess_id + ""

    # MIGRATION EN COURS VERS UN MIRROIR DE LA BUCKET
    os.system("gsutil cp -r " + folder_name + " " + bucket + json_["Output_file"])
    folder_name = folder_name + "/results"

    # done should use with statement
    with open(folder_name + "/env.txt", "w") as fp:
        fp.write("0")

    # LECTURE DU DF
    df = fast_read_excel(
        data + json_["input_df"]
    )  
    plogger.log("applying filters for " + sess_id)

    # FILTRE PAR BMI et AGE si l'utilisateur l'a indique
    log_msg = "START applying filters for {}".format(sess_id)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    if BMI is not None:
        try:
            if BMI[0] >= 0:
                df = df.loc[np.where(df["BMI"] >= BMI[0])[0]]
            if BMI[1] >= 0:
                df = df.iloc[np.where(df["BMI"] <= BMI[1])[0]]
        except Exception as e:
            log_msg = (
                "WARNING empty DataFrame after applying BMI filters for {} - {}".format(
                    sess_id, e
                )
            )
            log_msg = plogger.create_log_message(
                log_msg, __file__, EXE.__name__, "WARNING"
            )
            plogger.log(
                message=log_msg, severity="WARNING", logger_name=LOGGER_NAME_VAL
            )
            env_[sess_id] = "Error: null BMI range"
            return

    if Age is not None:
        try:
            if Age[0] >= 0:
                dfT = df.loc[np.where(df["age"] > Age[0])[0]]
                df = dfT
            if Age[1] >= 0:
                dfT = df.iloc[np.where(df["age"] < Age[1])[0]]
                df = dfT
        except Exception as e:
            log_msg = (
                "WARNING empty DataFrame after applying Age filters for {} - {}".format(
                    sess_id, e
                )
            )
            log_msg = plogger.create_log_message(
                log_msg, __file__, EXE.__name__, "WARNING"
            )
            plogger.log(
                message=log_msg, severity="WARNING", logger_name=LOGGER_NAME_VAL
            )
            env_[sess_id] = "Error: null Age range"
            return
    # Passage du mm au cm

    df = df[json_["columns"]] / 10.0

    # SELECTION DE L'ALGO BBSS ou FFSS, si input n'existe pas utiliser FFSS
    BBSS = json_["BBSS"]

    # Pre selection des attributs, l'input utilisateur sera considere si ce dernier existe
    columns = json_["columns"]

    # Lecture des modeles, si n'existe pas exit
    poly_dic = {}
    pred_dic = {}

    log_msg = "START reading models in session {}".format(sess_id)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    for c in range(len(columns)):
        try:
            pred_dic["n" + str(c + 1)] = load_joblib(
                model_path + "/" + json_["category"] + "/std/" + columns[c] + "_s_v3",
                False,
            )
            pred_dic["p" + str(c + 1)] = load_joblib(
                model_path + "/" + json_["category"] + "/std/" + columns[c] + "_l_v3",
                False,
            )
            poly_dic["n" + str(c + 1)] = load_joblib(
                model_path
                + "/"
                + json_["category"]
                + "/std/poly_"
                + columns[c]
                + "_s_v3",
                False,
            )
            poly_dic["p" + str(c + 1)] = load_joblib(
                model_path
                + "/"
                + json_["category"]
                + "/std/poly_"
                + columns[c]
                + "_l_v3",
                False,
            )
        except:
            None
    # Parametre infere par une fonction
    max_ = Gen_helper_functions.get_max(df.values, pred_dic, poly_dic)
    # initialisation des tailles
    centroides, init_classes = SOM.get_SOM(
        df, json_["SOM"][0], json_["SOM"][1], json_["K"]
    )
    # print("MAP SIZE : "+centroides.shape)
    # FICHIER RESULTATS
    result_file = folder_name + "/results.txt"
    # todo pourquoi utiliser codecs.open?
    file_write_res = codecs.open(result_file, "w")

    file_write_res.write("manh_mean,perc\n")
    log_msg = "START launching threads from gen_algorithm for {}".format(sess_id)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    # EXECUTION D'un instance de l'algo génétique soit en multi threading ou en sequentiel. sequentiel pour les tests
    with multi_proc_helper.NonDaemonPool(10) as p:
        for num_classes in range(json_["minClasses"], json_["maxClasses"]):
            gen_algorithm.exec_p(
                env_,
                sess_id,
                100 / (json_["numGen"] * (json_["maxClasses"] - json_["minClasses"])),
                df,
                folder_name,
                json_["Output_file"],
                num_classes + 1,
                centroides,
                init_classes,
                columns,
                BBSS,
                max_,
                pred_dic,
                poly_dic,
                json_["numGen"],
                json_["halfPopSize"],
                json_["seed"],
            )

        p.close()
        p.join()
    # FIN DE L'EXECUTION, EN INFORMER L'UTILISATEUR PAR UN OK
    log_msg = "END launching threads from gen_algorithm for {}".format(sess_id)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    # done should use with statement
    with open(folder_name + "/env.txt", "w") as fp:
        fp.write("Ok")

    # COPIE DES RESULTATS

    try:
        zipf = zipfile.ZipFile(
            pref + "output/sizing-system/" + sess_id + ".zip", "w", zipfile.ZIP_DEFLATED
        )
        zipdir(folder_name, zipf)
        zipf.close()
        os.system(
            "cp "
            + pref
            + "output/sizing-system/"
            + sess_id
            + ".zip "
            + folder_name
            + "/res_gen.zip"
        )
        os.system(
            "gsutil cp "
            + pref
            + "output/sizing-system/"
            + sess_id
            + ".zip "
            + bucket
            + "output/sizing-system/result/genetique/zip/  "
        )

    except Exception as e:
        log_msg = "Failed to zip results for genetic algorithm {} {}".format(
            sess_id, str(e)
        )
        log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "ERROR")
        plogger.log(message=log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
    list_files = get_list_local_files(folder_name)
    os.system(
        "gsutil cp " + folder_name + "/env.txt " + bucket + json_["Output_file"] + "/"
    )
    upload_blobs(storage_client, BUCKET_NAME, list_files, json_["Output_file"])
    message = "END genetic algorithm EXE({params}) -> {returns}".format(
        params=log_params, returns=list_files
    )
    message = plogger.create_log_message(message, __file__, EXE.__name__, "INFO")
    plogger.log(message=message, severity="INFO", logger_name=LOGGER_NAME_VAL)
