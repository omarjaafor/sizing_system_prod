# Ce fichier genere un taillant en utilisant l'algorithme de quadrillage.
# Il commence par creer les fichier dont il a besoin.
# Il genere ensuite plusieurs taillant en appelante la fonction optitaill en variant ses parametre. Cette fonction genere un taillant.
# Ensuite il selectionne le meilleur taillant et genere des metadonnees sur ce dernier. Il effectue cela en listant des modeles puis en appelant get_cats
# et plot_grille.


import codecs
import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.patches as mpatches

# from bokeh.palettes import Spectral4
# from bokeh.plotting import figure, output_file, show
import os
import io
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import copy
from sklearn.neighbors import NearestNeighbors
import math
import copy
import scipy
import igraph as ig
import multiprocessing
from multiprocessing import Pool, Manager
import json
import xlsx2csv as xls
import sys
import argparse
import joblib
from lib.constants import *
from lib import plogger
from lib.gcp_function import (
    get_list_local_files,
    upload_blobs,
    download_blob,
    upload_blob,
    download_blobs,
)
from google.cloud import storage
import os
from os.path import basename
import zipfile

plogger.create_logger(os.getenv(LOGGER_NAME, LOGGER_NAME_VAL))

pref = "/"

model_path = pref + "output/sizing-system/models"
data = pref + "input/sizing-system/data/"
storage_client = storage.Client()
global pas
pas = 1 / (10 * 10)

global manager
manager = Manager()


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), basename(os.path.join(root, file)))


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


# manhattan dist
# permet de fabriquer le grid
def get_min_max(Pcm, Qcm, df):
    min_x = math.floor(df[df.columns[0]].min())
    max_x = math.ceil(df[df.columns[0]].max())
    min_y = math.floor(df[df.columns[1]].min())
    max_y = math.ceil(df[df.columns[1]].max())
    P = math.ceil((max_x - min_x) / float(Pcm))
    Q = math.ceil((max_y - min_y) / float(Qcm))
    prange = P * Pcm
    qrange = Q * Qcm
    min_x2 = min_x + (max_x - min_x) / float(2) - prange / float(2)
    max_x2 = min_x + (max_x - min_x) / float(2) + prange / float(2)
    min_y2 = min_y + (max_y - min_y) / float(2) - qrange / float(2)
    max_y2 = min_y + (max_y - min_y) / float(2) + qrange / float(2)
    return min_x2, max_x2, min_y2, max_y2, P, Q


# Manhattan dist
def dist_func(a, b):
    return float(np.abs(a - b).sum())


# cree un grid, utilisee en amont de la generation du taillant (avant optitaill)
def get_max(df, pred_dic, poly_dic):
    dfvn = df.sum(axis=0)
    a0 = np.min(df[:, 0])
    a1 = np.max(df[:, 0])
    b0 = np.min(df[:, 1])
    b1 = np.max(df[:, 1])

    if "p1" in pred_dic:
        X_tr = poly_dic["p1"].fit_transform(a0.reshape(-1, 1))
        X_tr2 = poly_dic["n1"].fit_transform(a0.reshape(-1, 1))
        aO_pred = max(pred_dic["p1"].predict(X_tr), pred_dic["n1"].predict(X_tr2))
        X_tr = poly_dic["p1"].fit_transform(a1.reshape(-1, 1))
        X_tr2 = poly_dic["n1"].fit_transform(a1.reshape(-1, 1))
        a1_pred = max(pred_dic["p1"].predict(X_tr), pred_dic["n1"].predict(X_tr2))
    else:
        aO_pred = 2 * 12
        a1_pred = 2 * 14

    if "p2" is pred_dic:
        X_tr = poly_dic["p2"].fit_transform(b0.reshape(-1, 1))
        X_tr2 = poly_dic["n2"].fit_transform(b0.reshape(-1, 1))
        b0_pred = max(pred_dic["p2"].predict(X_tr), pred_dic["n2"].predict(X_tr2))
        X_tr = poly_dic["p2"].fit_transform(b1.reshape(-1, 1))
        X_tr2 = poly_dic["n2"].fit_transform(b1.reshape(-1, 1))
        b1_pred = max(pred_dic["p2"].predict(X_tr), pred_dic["n2"].predict(X_tr2))
    else:
        b0_pred = 2 * 12
        b1_pred = 2 * 14

    # a0=(1/(aO_pred+5))*math.pow(2*aO_pred,2)/dfvn[0]
    # a1=(1/(a1_pred+5))*math.pow(2*a1_pred,2)/dfvn[0]
    # b0=(1/(b0_pred+5))*math.pow(2*b1_pred,2)/dfvn[1]
    # b1=(1/(b1_pred+5))*math.pow(2*b1_pred,2)/dfvn[1]
    a0 = (1 / (5 + aO_pred)) * math.pow(a0 - aO_pred, 2) / dfvn[0]
    a1 = (1 / (5 + a1_pred)) * math.pow(a1 - a1_pred, 2) / dfvn[0]
    b0 = (1 / (5 + b0_pred)) * math.pow(b0 - b0_pred, 2) / dfvn[1]
    b1 = (1 / (5 + b1_pred)) * math.pow(b1 - b1_pred, 2) / dfvn[1]

    return math.sqrt(max(a0, a1) + max(b0, b1))


# fonction qui genere des meta-donnees apres la generation du taillant (apres optitaill)
def get_cats(df, centroides, alpha_, beta_, alpha_2, beta_2, pred_dic, poly_dic, max_):
    dfv = df.values
    dfvn = dfv.sum(axis=0)
    dfv_2 = np.zeros([dfv.shape[0], centroides.shape[0]])
    dfv_2[:] = 0

    knn = NearestNeighbors(n_neighbors=centroides.shape[0], metric="manhattan")
    knn.fit(centroides)
    distances, indices = knn.kneighbors(dfv)
    indices = np.zeros([dfv.shape[0], centroides.shape[0]])
    for i in range(centroides.shape[0]):
        indices[:, i] = i
    d_ = np.zeros([dfv.shape[0], centroides.shape[0]], dtype=float)
    d_m = np.zeros([dfv.shape[0], centroides.shape[0]], dtype=float)

    d_norm = np.zeros([dfv.shape[0], centroides.shape[0]], dtype=float)
    cat1 = {}
    cat2 = {}
    diff_m = {}
    cat1All = set()
    musp = {}
    musn = {}
    sigmasp = {}
    sigmasn = {}
    sigmas = {}
    mus = {}
    dist_ = {}
    norms_ = {}
    deg_f = {}

    for i in range(centroides.shape[0]):
        mus[i] = [centroides[i, 0], centroides[i, 1]]
        if "p1" not in poly_dic:
            ap = 2 * 12
            an = 2 * 14
        else:
            X_tr = poly_dic["p1"].fit_transform(mus[i][0].reshape(-1, 1))
            ap = pred_dic["p1"].predict(X_tr)

            X_tr = poly_dic["n1"].fit_transform(mus[i][0].reshape(-1, 1))
            an = pred_dic["n1"].predict(X_tr)
        if "p2" not in poly_dic:
            bp = 2 * 12
            bn = 2 * 14
        else:
            X_tr = poly_dic["p2"].fit_transform(mus[i][1].reshape(-1, 1))
            bp = pred_dic["p2"].predict(X_tr)

            X_tr = poly_dic["n2"].fit_transform(mus[i][1].reshape(-1, 1))
            bn = pred_dic["n2"].predict(X_tr)

        sigmasp[i] = [ap, bp]
        sigmasn[i] = [an, bn]

        norms_[i] = [scipy.stats.norm(), scipy.stats.norm()]
        print(sigmasn[i])
        norms_[i] = [scipy.stats.norm(), scipy.stats.norm()]
        diff_m[i] = 0  # max(0,pred_means.predict(mus[i][1].reshape(1, -1))[0])

    for j in range(dfv.shape[0]):
        for k in range(centroides.shape[0]):
            if dfv[j, 0] >= mus[k][0]:
                a = (
                    (1 / (sigmasp[k][0] + 5))
                    * math.pow((dfv[j, 0] - mus[k][0]), 2)
                    / dfvn[0]
                )
                # a=(1/(1+5))*math.pow((dfv[j,0]-mus[indices[j,k]][0]),2)/dfvn[0]
                d_norm[j, k] = norms_[k][1].pdf(
                    abs(dfv[j, 0] - mus[k][0]) / sigmasp[k][0]
                )
                aa = abs(dfv[j, 0] - mus[k][0]) / sigmasp[k][0]
                if aa > dfv_2[j, k]:
                    dfv_2[j, k] = aa
            else:
                a = (
                    (1 / (sigmasn[k][0] + 5))
                    * math.pow((dfv[j, 0] - mus[k][0]), 2)
                    / dfvn[0]
                )
                # a=(1/(1+5))*math.pow((dfv[j,0]-mus[indices[j,k]][0]),2)/dfvn[0]
                d_norm[j, k] = norms_[k][1].pdf(
                    abs(dfv[j, 0] - mus[k][0]) / sigmasn[k][0]
                )
                aa = abs(dfv[j, 0] - mus[k][0]) / sigmasn[k][0]
                if aa > dfv_2[j, k]:
                    dfv_2[j, k] = aa
            if dfv[j, 1] >= mus[k][1]:
                b = (
                    (1 / (sigmasp[k][1] + 5))
                    * math.pow((dfv[j, 1] - mus[k][1]), 2)
                    / dfvn[1]
                )
                # b=(1/(1+5))*math.pow((dfv[j,1]-mus[indices[j,k]][1]),2)/dfvn[1]
                aa = abs(dfv[j, 1] - mus[k][1]) / sigmasp[k][1]
                d_norm[j, k] = d_norm[j, k] * norms_[k][1].pdf(
                    abs(dfv[j, 1] - mus[k][1]) / sigmasp[k][1]
                )
                if aa > dfv_2[j, k]:
                    dfv_2[j, k] = aa

            else:
                b = (
                    (1 / (sigmasn[k][1] + 5))
                    * math.pow((dfv[j, 1] - mus[k][1]), 2)
                    / dfvn[1]
                )
                # b=(1/(1+5))*math.pow((dfv[j,1]-mus[indices[j,k]][1]),2)/dfvn[1]
                aa = abs(dfv[j, 1] - mus[k][1]) / sigmasn[k][1]
                d_norm[j, k] = d_norm[j, k] * norms_[k][1].pdf(
                    abs(dfv[j, 1] - mus[k][1]) / sigmasn[k][1]
                )
                if aa > dfv_2[j, k]:
                    dfv_2[j, k] = aa
            distances[j, k] = math.sqrt(a + b)

            d_[j, k] = math.sqrt(
                math.pow(dfv[j, 1] - mus[k][1], 2) + math.pow(dfv[j, 0] - mus[k][0], 2)
            )
            d_m[j, k] = abs(dfv[j, 1] - mus[k][1]) + abs(dfv[j, 0] - mus[k][0])

        indices[j, :] = indices[j, np.argsort(distances[j, :])]
        d_[j, :] = d_[j, np.argsort(d_[j, :])]
        d_m[j, :] = d_m[j, np.argsort(d_m[j, :])]
        dfv_2[j, :] = dfv_2[j, np.argsort(dfv_2[j, :])]
        d_norm[j, :] = d_norm[j, np.argsort(-d_norm[j, :])]
        d = distances[j, np.argsort(distances[j, :])]

        distances[j, :] = d

    for iter_ in range(centroides.shape[0]):
        i = 0
        for s in range(centroides.shape[0]):
            if s not in cat1:
                cat1[s] = set()
            if True:
                inside_box = set(np.where(dfv_2[:, iter_] < 2)[0])
                # inside_box=set()
            cat1[s] = cat1[s] | (
                (set(np.where(indices[:, iter_] == s)[0]) & inside_box) - cat1All
            )
            cat1All = cat1All | cat1[s]
            # cat3=set(np.arange(dfv.shape[0]))-cat1-cat2
            i += 1
    fitness = 0

    for k, vals in cat1.items():
        for v in vals:
            fitness += distances[v, 0]
    fitness_2 = 0
    for k in range(df.shape[0]):
        fitness_2 += d_norm[k, 0]
        if k not in cat1All:
            fitness += max_
    cat3 = set(np.arange(dfv.shape[0])) - cat1All

    return (
        len(cat1All) / float(dfv.shape[0]),
        fitness,
        fitness_2,
        d_m[:, 0].sum(),
        d_[:, 0].sum(),
    )


# helper pour optitaill
def partition_df(
    dfx, dfy, prange, qrange, min_x, min_y, max_x, max_y, alpha=10, beta=10
):
    cat1 = {}
    cat2 = {}
    graph_builder = {}
    i = 0
    blocksize_x = (max_x - min_x) / float(prange)
    blocksize_y = (max_y - min_y) / float(qrange)
    centroides = []
    for p in range(prange):
        a = dfx.loc[
            (dfx.index < min_x + (p + 1) * blocksize_x)
            & (dfx.index >= min_x + (p) * blocksize_x)
        ]

        a = set(list(a[a.columns[1]]))
        a2 = dfx.loc[
            (dfx.index < min_x + (p) * blocksize_x)
            & (dfx.index >= min_x + (p + 0.5) * blocksize_x - alpha)
        ]

        a2 = set(list(a2[a2.columns[1]]))
        for q in range(qrange):
            if True:
                c = [0, 0]
                c[0] = min_x + (p + 0.5) * blocksize_x
                b = dfy.loc[
                    (dfy.index < min_y + (q + 1) * blocksize_y)
                    & (dfy.index >= min_y + (q) * blocksize_y)
                ]
                b = set(list(b[b.columns[1]]))
                cat1[i] = a & b
                c[1] = min_y + (q + 0.5) * blocksize_y
                b2 = dfy.loc[
                    (dfy.index < min_y + (q) * blocksize_y)
                    & (dfy.index >= min_y + ((q + 0.5) * blocksize_y) - beta)
                ]
                b2 = set(list(b2[b2.columns[1]]))
                cat2[i] = (a2 & b2) | (a & b2) | (a2 & b)
                graph_builder[i] = [p, q]
                cat2[i] = cat2[i] - cat1[i]
                centroides.append(c)
                i += 1

    return cat1, cat2, graph_builder, centroides


# helper pour optitaill
def compute_fit_block(df, cat1, cat2, centroides, ind, alpha_, beta_):
    returnVal = 0

    for c in cat1[ind]:
        returnVal += dist_func(
            centroides[ind, :], df.iloc[c][[df.columns[0], df.columns[1]]]
        )
    for c in cat2[ind]:
        returnVal += dist_func(
            centroides[ind, :], df.iloc[c][[df.columns[0], df.columns[1]]]
        )

    for di in range(df.shape[0]):
        if di not in cat1[ind] and di not in cat2[ind]:
            # returnVal+=dist_func(centroides[ind,:],df.iloc[di][[df.columns[0],df.columns[1]]])+alpha_+beta_
            returnVal += alpha_ + beta_
    return returnVal


# helper pour optitaill
def check_conn(graph_builder, i, j):
    if (abs(graph_builder[i][0] - graph_builder[j][0]) <= 1) and (
        abs(graph_builder[i][1] - graph_builder[j][1]) <= 1
    ):
        return True
    else:
        False


# genere des metadonnees apres la generation du taillant
def plot_grille(
    df,
    centroides,
    prange,
    qrange,
    min_x,
    min_y,
    max_x,
    max_y,
    fit,
    selected,
    cl,
    cat1,
    cat2,
    it,
    pred_dic,
    poly_dic,
    max_,
    FOLDER_NAME,
):
    blocksize_x = (max_x - min_x) / float(prange)
    blocksize_y = (max_y - min_y) / float(qrange)
    fig = plt.figure()
    # plt.grid()
    ax = fig.add_subplot(111)
    ax.axis("equal")
    col_ind = np.array(["r"] * df.shape[0])
    col_ind_int = np.array([0] * df.shape[0])
    cat1set = set()
    for s in selected:
        for c in cat1[s]:
            cat1set.add(c)
    cat2set = set()
    for s in selected:
        for c in cat2[s]:
            cat2set.add(c)
    cat2set = cat2set - cat1set
    cat1set = np.array(list(cat1set), dtype=int)
    cat2set = np.array(list(cat2set), dtype=int)
    col_ind[cat1set] = "y"
    col_ind_int[cat1set] = 2
    col_ind[cat2set] = "b"
    col_ind_int[cat2set] = 1
    knn = NearestNeighbors(n_neighbors=1, metric="manhattan")
    knn.fit(centroides[selected, :])
    distances, indices = knn.kneighbors(df[[df.columns[0], df.columns[1]]].values)
    col_zeros = np.where(col_ind == 0)[0]
    col_ind_int = indices[:, 0]
    col_ind_int[col_zeros] = 0
    np.savetxt(FOLDER_NAME + "/df_affectation" + "." + str(it) + ".csv", col_ind_int)

    ax.scatter(df[df.columns[0]], df[df.columns[1]], s=10, c=col_ind)
    # ax2 = fig.add_subplot(211)
    x = []
    y = []
    order_ = []
    k = 0
    for v in centroides[selected, :]:
        x.append(v[0])
        y.append(v[1])

        order_.append(k)
        k += 1
    x = np.array(x)

    y = np.array(y)
    ax.scatter(x, y, 25, c="purple")

    f_ = []
    for o in order_:
        f_.append(fit[o])
    f_ = np.array(f_)
    f_ = f_ / f_.sum()

    col = ["r"] * centroides.shape[0]
    col = np.array(col)
    col[selected] = "g"
    # ax.scatter(x,y,s=f_*2000,c=col)

    # set your ticks manually
    plt.xticks(
        [min_x + k * blocksize_x for k in range(prange + 1)], rotation="vertical"
    )
    ax.yaxis.set_ticks([min_y + k * blocksize_y for k in range(qrange + 1)])
    ax.grid(True)
    # plt.savefig(FOLDER_NAME + "/" + str(it) + '.pdf', format="pdf")
    # plt.savefig(FOLDER_NAME + "/" + str(it) + '.png')
    plt.show()
    if False:
        with open(FOLDER_NAME + "/cat1." + str(it) + ".txt", "w") as file:
            for c in cat1:
                file.write(str(c) + "\n")
        with open(FOLDER_NAME + "/cat2." + str(it) + ".txt", "w") as file:
            for c in cat2:
                file.write(str(c) + "\n")
    # return ax

    return get_cats(
        df.iloc[:, [0, 1]],
        centroides[selected, :],
        10,
        6,
        5,
        3,
        pred_dic,
        poly_dic,
        max_,
    )


# helper pour optitaill
def update_fit_4(
    next_, fit, n, cat1, cat2, df, selection, gains, centroides, alpha_, beta_
):
    k = 0
    fit[:] = 0

    for c in cat1[next_]:
        gains[c] = dist_func(
            centroides[next_], df.iloc[c][[df.columns[0], df.columns[1]]]
        )

    for c in range(df.shape[0]):
        if c not in cat1[next_]:
            if c in cat2[next_]:
                d = dist_func(
                    centroides[next_], df.iloc[c][[df.columns[0], df.columns[1]]]
                )
            else:
                d = alpha_ + beta_
            if c not in gains:
                gains[c] = d
            elif gains[c] > d:
                gains[c] = d

    for k in n - set(selection):
        for c in cat1[k]:
            if c in gains:
                fit[k] += (
                    dist_func(centroides[k], df.iloc[c][[df.columns[0], df.columns[1]]])
                    - gains[c]
                )
            else:
                fit[k] += dist_func(
                    centroides[k], df.iloc[c][[df.columns[0], df.columns[1]]]
                )
        for c in range(df.shape[0]):
            if c not in cat1[k]:
                if c in cat2[k]:
                    d = dist_func(
                        centroides[k], df.iloc[c][[df.columns[0], df.columns[1]]]
                    )
                else:
                    d = alpha_ + beta_
                if c in gains:
                    if gains[c] > d:
                        fit[k] += d - gains[c]

    return fit, gains


# helper pour optitaill
def compute_fit_partition2(gains):
    fit_ = 0
    for k, v in gains.items():
        fit_ += v
    return fit_


# genere un taillant avec les parametres Pcm et Qcm (on la varie pour generer plusieurs taillant et on selection ensuite le meilleur)
def optitaill(
    Pcm,
    Qcm,
    alpha_,
    beta_,
    df,
    dfx,
    dfy,
    minclasses,
    maxclasses,
    iteration_P,
    iterationQ,
    quality,
    selection_dic,
    gains_com,
    results,
    env_,
    pas,
    sess_id,
    outfile,
):
    iter_ = 0
    print("OPTITAILL")
    min_x, max_x, min_y, max_y, P, Q = get_min_max(Pcm, Qcm, df)
    numComs = maxclasses  # -minclasses
    if numComs > P * Q:
        numComs = P * Q
    print("min_x,max_x,min_y,max_y,P,Q,Pcm,Qcm,numcom")
    print(min_x, max_x, min_y, max_y, P, Q, Pcm, Qcm, numComs)

    iteration_P.append(Pcm)
    iterationQ.append(Qcm)

    cat1, cat2, graph_builder, centroides = partition_df(
        dfx, dfy, P, Q, min_x, min_y, max_x, max_y
    )
    centroides = np.array(centroides)
    print("ETAPE 1")
    # FIT
    fit = {}
    k = 0  # minclasses-1
    for v in centroides:
        fit[k] = compute_fit_block(df, cat1, cat2, centroides, k, alpha_, beta_)
        k += 1

    # MAKE SELECTIION

    unique = np.unique(list(graph_builder.keys()))

    edgelist = []
    f_ = np.zeros(np.max(unique) + 1)
    print("3")
    for u1 in unique:
        if u1 in fit:
            f_[u1] = fit[u1]
            for u2 in unique:
                if u2 in fit and u1 != u2 and check_conn(graph_builder, u1, u2):
                    edgelist.append([u1, u2])

    vcount = np.max(unique)
    g = ig.Graph(vcount, edgelist)
    print("ETAPE 2")
    start_ = np.argmin(f_)
    selected = []
    selected.append(start_)

    n = set(g.neighbors(start_, mode="all")) - {start_}
    nselection = set()
    nselection2 = set()
    gains = {}
    print("4")
    f_, gains = update_fit_4(
        start_, f_, n, cat1, cat2, df, selected, gains, centroides, alpha_, beta_
    )
    ga = compute_fit_partition2(gains)
    if 1 not in gains_com:
        gains_com[1] = manager.list([ga, Pcm, Qcm])
        selection_dic[1] = manager.list(copy.deepcopy(selected))

    elif ga < gains_com[1][0]:
        gains_com[1] = manager.list([ga, Pcm, Qcm])
        selection_dic[1] = manager.list(copy.deepcopy(selected))
    r = []
    r.append([ga, 1])
    print("5")
    print("ETAPE 3")
    for k in range(maxclasses):
        a = np.array(list(n))
        next_ = a[np.argmin(f_[a])]
        selected.append(next_)
        n = n | set(g.neighbors(next_, mode="all"))
        n = n - set(selected)
        f_, gains = update_fit_4(
            next_, f_, n, cat1, cat2, df, selected, gains, centroides, alpha_, beta_
        )
        ga = compute_fit_partition2(gains)
        r.append([ga, k + 2])

        if (k + 2) not in gains_com:
            gains_com[k + 2] = manager.list([ga, Pcm, Qcm])
            selection_dic[k + 2] = manager.list(copy.deepcopy(selected))

        elif ga < gains_com[k + 2][0]:
            gains_com[k + 2] = manager.list([ga, Pcm, Qcm])
            selection_dic[k + 2] = manager.list(copy.deepcopy(selected))
        env_[sess_id] += pas / float(maxclasses)
        print("update session:", pas / float(maxclasses))
    print("FINISHED")
    try:
        os.mkdir(pref + "output/sizing-system/" + sess_id)
    except:
        None
    fp = open(pref + "output/sizing-system/" + sess_id + "/env.txt", "w")
    print("ETAPE 4")
    fp.write(str(env_[sess_id]))
    print(env_[sess_id])
    fp.close()
    print("gsutil 1")
    plogger.log("(GBSS,{},{})".format(sess_id, env_[sess_id]))
    print("(GBSS,{},{})".format(sess_id, env_[sess_id]))
    # os.system("gsutil cp " + pref + "output/sizing-system/" + sess_id + "/env.txt " + bucket+outfile)
    upload_blob(
        storage_client,
        BUCKET_NAME,
        pref + "output/sizing-system/" + sess_id + "/env.txt",
        outfile + "/env.txt",
    )
    ga = compute_fit_partition2(gains)
    results[str(Qcm) + "-" + str(Pcm)] = manager.list(r)
    quality.append(ga)

    iter_ += 1
    del g
    # env_["session_"+SESSION_ID]+=0.6*pas


def EXE(sess_id):
    folder_name = "/input/sizing-system/" + sess_id
    log_params = {"sess_id": sess_id}
    log_msg = "START quadrillage EXE({params})".format(params=log_params)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    user_input_json = "/input/sizing-system/setting/input_" + sess_id + ".json"
    log_msg = "DOWNLOAD user input {}".format(user_input_json)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    # os.system(
    #    "gsutil cp -r " + bucket + "input/sizing-system/setting/input_" + sess_id + ".json " + pref + "input/sizing-system/setting/ ")
    # dd = download_blob(storage_client, BUCKET_NAME, 'input/sizing-system/setting/input_' + sess_id + '.json',
    #                   pref + 'input/sizing-system/setting/input_' + sess_id + '.json')
    # print("gsutil cp -r " + bucket + "input/sizing-system/setting/input_" + sess_id + ".json " + pref + "input/sizing-system/setting/ ")
    if not os.path.exists(pref + user_input_json):
        log_msg = "FAILED json file does not exist: {}".format(pref + user_input_json)
        log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "ERROR")
        plogger.log(log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)
        raise FileNotFoundError(
            "json file does not exist: {}".format(pref + user_input_json)
        )
    with open(pref + user_input_json) as json_file:
        json_ = json.load(json_file)
    print("EXE ETAPE 1")
    alpha_, beta_ = json_["alpha_"], json_["beta_"]
    minclasses, maxclasses = json_["minclasses"], json_["maxclasses"]
    env_ = manager.dict()
    sess_id = json_["sess_id"]
    env_[sess_id] = 0

    BMI = json_["BMI"]
    if BMI is not None:
        BMI = np.array(BMI)
    Age = json_["Age"]
    if Age is not None:
        Age = np.array(Age)
    try:
        os.mkdir(pref + "output/sizing-system/" + sess_id)
    except:
        None
    try:
        os.mkdir(pref + "output/sizing-system/" + sess_id + "/results")
    except:
        None

    with open(pref + "output/sizing-system/" + sess_id + "/env.txt", "w") as fp:
        fp.write("0")

    folder_name = pref + "output/sizing-system/" + sess_id + ""
    print("EXE ETAPE 2")
    # os.system("gsutil cp -r " + folder_name + "/ " + bucket+json_["Output_file"])
    # list_files = get_list_local_files(folder_name)
    # upload_blobs(storage_client, BUCKET_NAME, list_files, json_["Output_file"])
    winnerP = manager.list()
    winnerQ = manager.list()
    results = manager.dict()
    iteration_P = manager.list()
    iterationQ = manager.list()
    quality = manager.list()
    selection_dic = manager.dict()
    qual_classes = manager.list()
    gains = manager.dict()
    gains_com = manager.dict()
    maxclasses = maxclasses - 1
    c = maxclasses  # -minclasses
    # TODO:  check if input_df is uploaded otherwise use a default path
    df = fast_read_excel(data + json_["input_df"])  # [json_["columns"]]/10.0
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
    print("EXE ETAPE 3")
    df = df[json_["columns"]] / 10.0
    df["id"] = np.arange(df.shape[0])
    dfx = df.set_index(df.columns[0])
    dfx = dfx.sort_index()
    dfy = df.set_index(df.columns[1])
    dfy = dfy.sort_index()
    print(int(multiprocessing.cpu_count() / 2))
    env_[sess_id] = 0
    pas = 100 / float(10 * 10 * maxclasses)
    print("BEFORE LAUNCHE")
    # with Pool(int(multiprocessing.cpu_count() / 2)) as p:
    # with Pool(2) as p:
    for Pcm in range(1, 3):
        for Qcm in range(1, 3):
            print("l optitaille")
            optitaill(
                Pcm,
                Qcm,
                alpha_,
                beta_,
                df,
                dfx,
                dfy,
                minclasses,
                maxclasses,
                iteration_P,
                iterationQ,
                quality,
                selection_dic,
                gains_com,
                results,
                env_,
                pas,
                sess_id,
                json_["Output_file"],
            )
            # p.apply_async(optitaill, args=(
            #     Pcm, Qcm, alpha_, beta_, df, dfx, dfy, minclasses, maxclasses, iteration_P, iterationQ, quality,
            #     selection_dic, gains_com, results, env_, pas, sess_id, json_["Output_file"]))
            # handler.get()
    # p.close()
    # p.join()
    print("closed")
    print("EXE ETAPE 4")
    log_msg = "START reading models in session {}".format(sess_id)
    log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "INFO")
    plogger.log(message=log_msg, severity="INFO", logger_name=LOGGER_NAME_VAL)

    poly_dic = {}
    pred_dic = {}
    columns = json_["columns"]
    try:
        pred_dic["n1"] = load_joblib(
            model_path + "/hauts/std/" + columns[0] + "_s_v3", False
        )
        pred_dic["p1"] = load_joblib(
            model_path + "/hauts/std/" + columns[0] + "_l_v3", False
        )
        poly_dic["n1"] = load_joblib(
            model_path + "/hauts/std//poly_" + columns[0] + "_s_v3", False
        )
        poly_dic["p1"] = load_joblib(
            model_path + "/hauts/std/poly_" + columns[0] + "_l_v3", False
        )
    except:
        None

    try:
        pred_dic["n2"] = load_joblib(
            model_path + "/hauts/std/" + columns[1] + "_s_v3", False
        )
        pred_dic["p2"] = load_joblib(
            model_path + "/hauts/std/" + columns[1] + "_l_v3", False
        )
        poly_dic["n2"] = load_joblib(
            model_path + "/hauts/std//poly_" + columns[1] + "_s_v3", False
        )
        poly_dic["p2"] = load_joblib(
            model_path + "/hauts/std/poly_" + columns[1] + "_l_v3", False
        )
    except:
        None
    # folder_name = folder_name + "/results"
    file_ = codecs.open(folder_name + "/meta.csv", "w")
    file_.write("nombre de tailles,Pcm,Qcm,P,Q,nombre de cases\n")
    max_ = get_max(df.values, pred_dic, poly_dic)
    it = 0  # minclasses-1
    for k, v in gains_com.items():
        it += 1
        print(v)

        cl = v[0]
        qual_classes.append(v[0])
        Pcm = v[1]
        Qcm = v[2]

        print("Pcm", Pcm)
        print("Qcm", Qcm)
        sel = selection_dic[k]
        min_x, max_x, min_y, max_y, P, Q = get_min_max(Pcm, Qcm, df)
        file_.write(
            str(it)
            + ","
            + str(Pcm)
            + ","
            + str(Qcm)
            + ","
            + str(P)
            + ","
            + str(Q)
            + ","
            + str(P * Q)
            + "\n"
        )

        print(min_x, max_x, min_y, max_y, P, Q)
        cat1, cat2, graph_builder, centroides = partition_df(
            dfx, dfy, P, Q, min_x, min_y, max_x, max_y
        )
        # centroides=get_centroides(cat1,df,P,Q,min_x,min_y,max_x,max_y)
        # FIT
        fit = {}
        cl = 0  # minclasses-1
        centroides = np.array(centroides)
        for v in centroides:
            fit[cl] = compute_fit_block(df, cat1, cat2, centroides, k, alpha_, beta_)
            cl += 1

        cov, boxfitness, stochasticfit, manhattanmin, euclidean = plot_grille(
            df,
            centroides,
            P,
            Q,
            min_x,
            min_y,
            max_x,
            max_y,
            fit,
            sel,
            cl,
            cat1,
            cat2,
            it,
            pred_dic,
            poly_dic,
            max_,
            folder_name,
        )
        print("cov,boxfitness,stochasticfit,manhattanmin,euclidean,v[0]")
        print(cov, boxfitness, stochasticfit, manhattanmin, euclidean, v[0])
        writer_results = codecs.open(folder_name + "/results.txt", "w")
        writer_results.write("manh_mean,perc\n")
        writer_results.write(str(manhattanmin / df.shape[0]) + "," + str(cov))
        writer_results.close()
        winnerP.append(Pcm)
        winnerQ.append(Qcm)
        print(centroides.shape)
        ids_ = np.zeros([centroides[sel, :].shape[0], 1])
        ids_[:, 0] = np.arange(centroides[sel, :].shape[0]) + 1
        print(ids_.shape)
        np.arange(centroides[sel, :].shape[0]) + 1
        print("centroides")
        print(centroides[sel, :])
        writer_results = codecs.open(
            folder_name + "/centroides." + str(it) + ".txt", "w"
        )
        writer_results.write(columns[0] + "," + columns[1])
        for i in range(len(sel)):
            writer_results.write(
                "\n"
                + str(i + 1)
                + ","
                + str(centroides[sel[i], 0])
                + ","
                + str(centroides[sel[i], 1])
            )
        # np.savetxt(folder_name + "/centroides." + str(it) + ".txt", np.concatenate((ids_, centroides[sel, :]), axis=1),
        #           delimiter=",", fmt='%.2e')

    file_.flush()
    file_.close()
    fig = plt.figure()
    fig.suptitle("Front de Pareto pour le modèle GBSS", fontsize=14)

    plt.plot(np.arange(len(qual_classes)) + 1, qual_classes)
    plt.xticks(np.arange(maxclasses) + 1)
    plt.xlabel("nombre de tailles", fontsize=10)
    plt.ylabel("FL", fontsize=10)
    # plt.savefig(folder_name + "/pareto.png")

    fig = plt.figure()
    fig.suptitle("Pas optimums par nombre de tailles", fontsize=14)
    plt.plot(np.arange(len(winnerP)) + 1, np.array(winnerP), color="r")
    plt.ylabel("Pas (cm)", fontsize=10)
    y_patch = mpatches.Patch(color="y", label="P (cm)")
    r_patch = mpatches.Patch(color="r", label="Q (cm)")
    plt.legend(handles=[y_patch, r_patch])
    # plt.xticks(np.arange(maxclasses)+1)
    plt.xlabel("nombre de tailles", fontsize=10)

    # plt.ylabel('P (cm)', fontsize=10)

    # plt.savefig(FOLDER_NAME+"/pcm.jpg")

    # fig = plt.figure()
    # fig.suptitle('Q (cm) par taille', fontsize=14)
    plt.plot(np.arange(len(winnerQ)) + 1, np.array(winnerQ), color="y")
    # plt.xticks(np.arange(maxclasses)+1)
    plt.yticks(np.arange(len(winnerQ)) + 1)
    # p#lt.xlabel('nombre de tailles', fontsize=10)
    # plt.ylabel('Q (cm)', fontsize=10)
    # plt.savefig(folder_name + "/pcm_qcm.png")
    # Pas optimum par taille
    # fig = plt.figure()
    # os.system("gsutil cp -r  " + folder_name + "/* " + bucket+json_["Output_file"])

    with open(folder_name + "/env.txt", "w") as fp:
        fp.write("Ok")
    try:
        # shutil.make_archive(folder_name+"/res_gen", 'zip', "/",folder_name)
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
        # make_archive(folder_name,pref + "output/sizing-system/" + sess_id +"/res_gen.zip")
    except Exception as e:
        log_msg = "Failed to zip results for genetic algorithm {} {}".format(
            sess_id, str(e)
        )
        log_msg = plogger.create_log_message(log_msg, __file__, EXE.__name__, "ERROR")
        plogger.log(message=log_msg, severity="ERROR", logger_name=LOGGER_NAME_VAL)

    list_files = get_list_local_files(folder_name)
    upload_blobs(storage_client, BUCKET_NAME, list_files, json_["Output_file"])

    # with open(pref + "output/sizing-system/" + sess_id + "/env.txt", 'w') as fp:
    #    fp.write("Ok")
    # os.system("gsutil cp  " + pref + "output/sizing-system/" + sess_id + "/env.txt " +bucket+ json_["Output_file"])

    message = "END quadrillage EXE({params}), results saved in {returns}".format(
        params=log_params, returns=folder_name
    )
    message = plogger.create_log_message(message, __file__, EXE.__name__, "INFO")
    plogger.log(message=message, severity="INFO", logger_name=LOGGER_NAME_VAL)
