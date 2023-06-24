# Ces fonctions sont utilisées dans BBSS. Ensemble des helpers functions pour l'algorithme de génération de tailles décrit dans : https://ieeexplore.ieee.org/abstract/document/9530202


import numpy as np
import math
import pandas as pd
import scipy
import copy
from sklearn.neighbors import NearestNeighbors
from matplotlib import colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import sys
import time
import json


# Decodage chromosome, génération des tailles à partir d'un chromosome
def get_max(df, pred_dic, poly_dic):
    dfvn = df.max(axis=0)

    aO_pred = {}
    # b={}

    for c in range(df.shape[1]):
        a0 = np.min(df[:, 0])
        a1 = np.max(df[:, 0])

        aO_pred[c] = [0, 0]

        if "p" + str(c + 1) in pred_dic:
            X_tr = poly_dic["p" + str(c + 1)].fit_transform(a0.reshape(-1, 1))
            X_tr2 = poly_dic["n" + str(c + 1)].fit_transform(a0.reshape(-1, 1))
            aO_pred[c][0] = max(
                pred_dic["p" + str(c + 1)].predict(X_tr),
                pred_dic["n" + str(c + 1)].predict(X_tr2),
            )
            X_tr = poly_dic["p" + str(c + 1)].fit_transform(a1.reshape(-1, 1))
            X_tr2 = poly_dic["n" + str(c + 1)].fit_transform(a1.reshape(-1, 1))
            aO_pred[c][1] = max(
                pred_dic["p" + str(c + 1)].predict(X_tr),
                pred_dic["n" + str(c + 1)].predict(X_tr2),
            )
        else:
            aO_pred[c][0] = 2 * 12
            aO_pred[c][1] = 2 * 14

    predicted_edge = []
    for c in range(df.shape[1]):
        a0 = (1 / (5 + aO_pred[c][0])) * math.pow(2 * aO_pred[c][0] / dfvn[0], 2)
        a1 = (1 / (5 + aO_pred[c][1])) * math.pow(2 * aO_pred[c][1] / dfvn[1], 2)
        predicted_edge.append(max(a0, a1))
    predicted_edge = np.array(predicted_edge)
    return math.sqrt(predicted_edge.sum())


# Calcule les fitness d'un taillant (BBSS & FFSS & Manh)
def compute_fitness(
    dfv, sol, centroides, set_box, max_, columns, BBSS, pred_dic, poly_dic
):
    # knn = NearestNeighbors(n_neighbors=sol.shape[0],metric="manhattan")
    # knn.fit(centroides[sol,:])
    # distances, indices = knn.kneighbors(dfv)

    distances = np.zeros([dfv.shape[0], sol.shape[0]])
    indices = np.zeros([dfv.shape[0], sol.shape[0]])
    for i in range(sol.shape[0]):
        indices[:, i] = i

    dfvn = dfv.max(axis=0)
    dfv_2 = np.zeros([dfv.shape[0], sol.shape[0]])
    d_norm = np.zeros([dfv.shape[0], sol.shape[0]], dtype=float)
    dfv_2[:] = 0
    musp = {}
    musn = {}
    sigmasp = {}
    sigmasn = {}
    sigmas = {}
    diff_m = {}
    mus = {}
    dist_ = {}
    norms_ = {}
    deg_f = {}

    for i in range(sol.shape[0]):
        mus[i] = []
        sigmasp[i] = []
        sigmasn[i] = []
        norms_[i] = []
        for c in range(dfv.shape[1]):
            mus[i].append(centroides[sol[i], c])
            if "p" + str(c + 1) not in poly_dic:
                sigmasp[i].append(2 * 12)
                sigmasn[i].append(2 * 14)
            else:
                X_tr = poly_dic["p" + str(c + 1)].fit_transform(
                    mus[i][0].reshape(-1, 1)
                )
                sigmasp[i].append(pred_dic["p" + str(c + 1)].predict(X_tr))

                X_tr = poly_dic["n" + str(c + 1)].fit_transform(
                    mus[i][0].reshape(-1, 1)
                )
                sigmasn[i].append(pred_dic["n" + str(c + 1)].predict(X_tr))

            norms_[i].append(scipy.stats.norm())

    for j in range(dfv.shape[0]):
        for k in range(sol.shape[0]):
            d_norm[j, k] = 1
            distances[j, k] = 0
            for c in range(dfv.shape[1]):
                if dfv[j, c] >= mus[k][c]:
                    a = (1 / (5 + sigmasp[k][c])) * math.pow(
                        (dfv[j, c] - mus[k][c]) / dfvn[c], 2
                    )
                    distances[j, k] += a
                    # a=(1/(1+5))*math.pow((dfv[j,0]-mus[indices[j,k]][0]),2)/dfvn[0]
                    d_norm[j, k] *= norms_[k][c].pdf(
                        abs(dfv[j, c] - mus[k][c]) / sigmasp[k][c]
                    )
                    aa = abs(dfv[j, c] - mus[k][c]) / sigmasp[k][c]
                    if dfv_2[j, k] < aa:
                        dfv_2[j, k] = aa
                else:
                    a = (1 / (5 + sigmasn[k][c])) * math.pow(
                        (dfv[j, c] - mus[k][c]) / dfvn[c], 2
                    )
                    # a=(1/(1+5))*math.pow((dfv[j,0]-mus[indices[j,k]][0]),2)/dfvn[0]
                    a = (1 / (5 + sigmasp[k][c])) * math.pow(
                        (dfv[j, c] - mus[k][c]) / dfvn[c], 2
                    )
                    distances[j, k] += a
                    d_norm[j, k] = norms_[k][c].pdf(
                        abs(dfv[j, c] - mus[k][c]) / sigmasn[k][c]
                    )
                    aa = abs(dfv[j, c] - mus[k][c]) / sigmasn[k][c]
                    if dfv_2[j, k] < aa:
                        dfv_2[j, k] = aa

            distances[j, k] = math.sqrt(distances[j, k])

        if BBSS:
            dist_sort = distances[j, :]
        else:
            dist_sort = -d_norm[j, :]

        indices[j, :] = indices[j, np.argsort(dist_sort)]
        dfv_2[j, :] = dfv_2[j, np.argsort(dist_sort)]
        d = distances[j, np.argsort(dist_sort)]
        d_norm[j, :] = d_norm[j, np.argsort(dist_sort)]
        distances[j, :] = d

    cat1 = {}
    cat2 = {}
    cat1All = set()

    for iter_ in range(sol.shape[0]):
        i = 0
        for s in sol:
            if s not in cat1:
                cat1[s] = set()
            inside_box = set(np.where(dfv_2[:, iter_] < 2)[0])

            cat1[s] = cat1[s] | (
                (inside_box & set(np.where(indices[:, iter_] == i)[0])) - cat1All
            )

            cat1All = cat1All | cat1[s]
            # cat3=set(np.arange(dfv.shape[0]))-cat1-cat2
            i += 1
    fitness = 0
    fitness_2 = 0
    for k, vals in cat1.items():
        for v in vals:
            fitness += distances[
                v, 0
            ]  # dist_func(dfv[v,:],centroides[k],alpha_,beta_,alpha_2,beta_2)

    for k in range(dfv.shape[0]):
        fitness_2 += d_norm[k, 0]
        if k not in cat1All:
            fitness += max_
    if BBSS:
        return fitness
    else:
        return -fitness_2


# A developper : pour l'instant le mutli-processing est utilisé dans BBSS.py. Si ce dernier utilise peu de processeurs (taillant à tailles limité), le
# multiprocessing devrait être utilisé ici.
def compute_fitness_multiproc(
    id_, dict_, df, sol, centroides, set_box, max_, columns, BBSS, pred_dic, poly_dic
):
    dict_[id_] = compute_fitness(
        df, sol, centroides, set_box, max_, columns, BBSS, pred_dic, poly_dic
    )


# decode les chromosomes
def get_cats(df, sol, centroides, set_box, max_, columns, BBSS, pred_dic, poly_dic):
    dfv = df.values
    dfvn = dfv.max(axis=0)
    dfv_2 = np.zeros([dfv.shape[0], sol.shape[0]])
    dfv_2[:] = 0

    knn = NearestNeighbors(n_neighbors=sol.shape[0], metric="manhattan")
    knn.fit(centroides[sol, :])
    distances, indices = knn.kneighbors(dfv)
    indices = np.zeros([dfv.shape[0], sol.shape[0]])
    for i in range(sol.shape[0]):
        indices[:, i] = i
    d_ = np.zeros([dfv.shape[0], sol.shape[0]], dtype=float)
    d_m = np.zeros([dfv.shape[0], sol.shape[0]], dtype=float)

    d_norm = np.zeros([dfv.shape[0], sol.shape[0]], dtype=float)
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

    for i in range(sol.shape[0]):
        mus[i] = []
        sigmasp[i] = []
        sigmasn[i] = []
        norms_[i] = []
        for c in range(dfv.shape[1]):
            mus[i].append(centroides[sol[i], c])
            if "p" + str(c + 1) not in poly_dic:
                sigmasp[i].append(2 * 12)
                sigmasn[i].append(2 * 14)
            else:
                X_tr = poly_dic["p" + str(c + 1)].fit_transform(
                    mus[i][0].reshape(-1, 1)
                )
                sigmasp[i].append(pred_dic["p" + str(c + 1)].predict(X_tr))

                X_tr = poly_dic["n" + str(c + 1)].fit_transform(
                    mus[i][0].reshape(-1, 1)
                )
                sigmasn[i].append(pred_dic["n" + str(c + 1)].predict(X_tr))

            norms_[i].append(scipy.stats.norm())

        diff_m[i] = 0  # max(0,pred_means.predict(mus[i][1].reshape(1, -1))[0])

    for j in range(dfv.shape[0]):
        for k in range(sol.shape[0]):
            d_norm[j, k] = 1
            distances[j, k] = 0
            d_[j, k] = 0
            d_m[j, k] = 0
            for c in range(dfv.shape[1]):
                if dfv[j, c] >= mus[k][c]:
                    a = (1 / (5 + sigmasp[k][c])) * math.pow(
                        (dfv[j, c] - mus[k][c]) / dfvn[c], 2
                    )
                    distances[j, k] += a
                    # a=(1/(1+5))*math.pow((dfv[j,0]-mus[indices[j,k]][0]),2)/dfvn[0]
                    d_norm[j, k] *= norms_[k][c].pdf(
                        abs(dfv[j, c] - mus[k][c]) / sigmasp[k][c]
                    )
                    aa = abs(dfv[j, c] - mus[k][c]) / sigmasp[k][c]
                    if dfv_2[j, k] < aa:
                        dfv_2[j, k] = aa
                else:
                    a = (1 / (5 + sigmasn[k][c])) * math.pow(
                        (dfv[j, c] - mus[k][c]) / dfvn[c], 2
                    )
                    # a=(1/(1+5))*math.pow((dfv[j,0]-mus[indices[j,k]][0]),2)/dfvn[0]
                    a = (1 / (5 + sigmasp[k][c])) * math.pow(
                        (dfv[j, c] - mus[k][c]) / dfvn[c], 2
                    )
                    distances[j, k] += a
                    d_norm[j, k] = norms_[k][c].pdf(
                        abs(dfv[j, c] - mus[k][c]) / sigmasn[k][c]
                    )
                    aa = abs(dfv[j, c] - mus[k][c]) / sigmasn[k][c]
                    if dfv_2[j, k] < aa:
                        dfv_2[j, k] = aa
                d_[j, k] += math.pow(dfv[j, c] - mus[k][c], 2)
                d_m[j, k] += abs(dfv[j, c] - mus[k][c])
            distances[j, k] = math.sqrt(distances[j, k])
            d_[j, k] = math.sqrt(d_[j, k])

        if BBSS:
            dist_sort = distances[j, :]
        else:
            dist_sort = -d_norm[j, :]

        indices[j, :] = indices[j, np.argsort(dist_sort)]
        d_[j, :] = d_[j, np.argsort(d_[j, :])]
        d_m[j, :] = d_m[j, np.argsort(d_m[j, :])]
        dfv_2[j, :] = dfv_2[j, np.argsort(dfv_2[j, :])]
        d_norm[j, :] = d_norm[j, np.argsort(-d_norm[j, :])]
        d = distances[j, np.argsort(distances[j, :])]

        distances[j, :] = d

    for iter_ in range(sol.shape[0]):
        i = 0
        for s in range(sol.shape[0]):
            if sol[s] not in cat1:
                cat1[sol[s]] = set()
            if True:
                inside_box = set(np.where(dfv_2[:, iter_] < 2)[0])
                # inside_box=set()
            cat1[sol[s]] = cat1[sol[s]] | (
                (set(np.where(indices[:, iter_] == s)[0]) & inside_box) - cat1All
            )
            cat1All = cat1All | cat1[sol[s]]
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
        -fitness_2,
        fitness,
        cat1All,
        cat1,
        len(cat3),
        len(cat3) / float(dfv.shape[0]),
        d_m[:, 0].sum(),
        d_[:, 0].sum(),
        sigmasp,
        sigmasn,
    )


# debugging
def plot_grille_rect(
    df,
    centroides,
    cat1All,
    cat1,
    it,
    folder_,
    sol,
    column_display_names,
    title,
    sigmasp,
    sigmasn,
):
    colors = [
        "#000000",
        "#8A2BE2",
        "#A52A2A",
        "#DEB887",
        "#5F9EA0",
        "#7FFF00",
        "#D2691E",
        "#FF7F50",
        "#6495ED",
        "#DC143C",
        "#00FFFF",
        "#00008B",
        "#008B8B",
        "#B8860B",
        "#006400",
        "#FFE4E1",
        "#DA70D6",
        "#008B8B",
        "#FFE4C4",
        "#FFEBCD",
        "#FF1493",
        "#0000FF",
    ]

    # colors =list(mcolors.CSS4_COLORS.keys())
    # plt.axis('off')
    fig = plt.figure()
    # plt.grid()
    fig.suptitle(
        "Modèle " + title + " avec " + str(sol.shape[0]) + " tailles", fontsize=14
    )
    plt.xlabel(column_display_names[0], fontsize=10)
    plt.ylabel(column_display_names[1], fontsize=10)
    ax = fig.add_subplot(111)
    ax.axis("equal")
    col_ind = np.array([colors[0]] * df.shape[0])

    # cat2=np.array(list(cat2),dtype=int)
    centr = list(cat1.keys())
    dict_centers = {}
    i = 0
    col = []
    for c in centr:
        dict_centers[c] = i
        i += 1
    for k, v in cat1.items():
        col_ind[np.array(list(v), dtype=int)] = colors[2 + dict_centers[k]]

    ax.scatter(df[df.columns[0]], df[df.columns[1]], s=5, c=col_ind)
    # ax2 = fig.add_subplot(211)
    x = []
    y = []
    order_ = []
    k = 0

    for v in range(sol.shape[0]):
        x.append(centroides[sol[v], 0])
        y.append(centroides[sol[v], 1])

        col.append(colors[1])
        k += 1

    x = np.array(x)

    y = np.array(y)

    col = np.array(col)

    ax.scatter(x, y, 25, c=col)
    print("X & Y")
    print(x)
    print(y)
    for i in range(len(x)):
        # Create a Rectangle patch
        print("sigmasn")
        print(sigmasn[i][0])
        print(sigmasn[i][1])
        print("sigmasp")
        print(sigmasp[i][0])
        print(sigmasp[i][1])

        print(x[i])
        print(y[i])
        print("===========")
        rect = patches.Rectangle(
            (x[i] - 2 * sigmasn[i][0], y[i] - 2 * sigmasn[i][1]),
            2 * sigmasn[i][0] + 2 * sigmasp[i][0],
            2 * sigmasn[i][1] + 2 * sigmasp[i][1],
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        # None
    plt.savefig(folder_ + "/" + title + "_" + str(it) + "_rect.pdf", format="pdf")
    plt.savefig(folder_ + "/" + title + "_" + str(it) + "_rect.png")
    plt.show()


# debugging
def plot_grille(
    df,
    centroides,
    cat1All,
    cat1,
    it,
    folder_,
    sol,
    column_display_names,
    title,
    json_writer,
):
    colors = [
        "#000000",
        "#8A2BE2",
        "#A52A2A",
        "#DEB887",
        "#5F9EA0",
        "#7FFF00",
        "#D2691E",
        "#FF7F50",
        "#6495ED",
        "#DC143C",
        "#00FFFF",
        "#00008B",
        "#008B8B",
        "#B8860B",
        "#006400",
        "#FFE4E1",
        "#DA70D6",
        "#008B8B",
        "#FFE4C4",
        "#FFEBCD",
        "#FF1493",
        "#0000FF",
    ]

    # colors =list(mcolors.CSS4_COLORS.keys())
    # plt.axis('off')
    fig = plt.figure()
    # plt.grid()
    fig.suptitle(
        "Modèle " + title + " avec " + str(sol.shape[0]) + " tailles", fontsize=14
    )
    plt.xlabel(column_display_names[0], fontsize=10)
    plt.ylabel(column_display_names[1], fontsize=10)
    ax = fig.add_subplot(111)
    ax.axis("equal")
    col_ind = np.array([colors[0]] * df.shape[0])
    sizes = np.array([0] * df.shape[0])

    # cat2=np.array(list(cat2),dtype=int)
    centr = list(cat1.keys())
    dict_centers = {}
    i = 0
    col = []
    for c in centr:
        dict_centers[c] = i
        i += 1

    for k, v in cat1.items():
        col_ind[np.array(list(v), dtype=int)] = colors[2 + dict_centers[k]]
        sizes[np.array(list(v), dtype=int)] = dict_centers[k] + 1
    ax.scatter(df[df.columns[0]], df[df.columns[1]], s=5, c=col_ind)

    np.savetxt(
        folder_ + "/df_affectation" + "." + title + "_" + str(it) + ".csv", sizes
    )
    # ax2 = fig.add_subplot(211)
    x = []
    y = []
    order_ = []
    k = 0

    for v in range(sol.shape[0]):
        x.append(centroides[sol[v], 0])
        y.append(centroides[sol[v], 1])

        col.append(colors[1])
        k += 1

    x = np.array(x)

    y = np.array(y)

    col = np.array(col)

    ax.scatter(x, y, 25, c=col)

    plt.savefig(folder_ + "/" + title + "_" + str(it) + ".pdf", format="pdf")
    plt.savefig(folder_ + "/" + title + "_" + str(it) + ".png")

    plt.show()


# combine 2 chromosomes (sl est le seed, )
def merge(sol1, sol2, sl):
    np.random.seed((333 * (sl)) % 10000)
    try:
        if sol1.shape[0] == 1:
            return sol1

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(centroides[sol1, :])
        distances, indices = knn.kneighbors(centroides[sol2, :])
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(centroides[sol2, :])
        distances_, indices_ = knn.kneighbors(centroides[sol1, :])

        sol_dic = {}
        for k in range(indices.shape[0]):
            if k == indices_[indices[k, 0], 0]:
                sol_dic[sol2[indices[k, 0]]] = sol1[k]

        new_sol = []
        i = 2
        np.random.seed((331 * (sl + i) + i) % 10000)
        shape_ = max(np.random.choice(sol2.shape[0] - 1, 1, replace=False), 1)
        order_1 = np.random.choice(sol2.shape[0], sol2.shape[0], replace=False)

        new_sol = []

        for i in range(shape_[0]):
            new_sol.append(sol1[order_1[i]])
        new_sol_set = set(new_sol)
        order_2 = np.random.choice(sol2.shape[0], sol2.shape[0], replace=False)
        i = 0
        while len(new_sol) < sol1.shape[0]:
            if sol2[order_2[i]] in sol_dic and sol_dic[sol2[order_2[i]]] in new_sol_set:
                None
            else:
                new_sol.append(sol2[order_2[i]])
            i += 1

        new_sol = np.array(new_sol)

        return new_sol
    except:
        return sol2


# mutation dans le cas où on cherche une seule taille
def mutation(sol, centroides, numclasses, a, sl, MutProb=0.05):
    np.random.seed((63 * sl + sl) % 10000)
    mutate_ = np.random.choice(2, 1)[0]
    if mutate_ == 0 or True:
        mut_ = 1

        new_sol = np.random.choice(centroides.shape[0], mut_, replace=False)

        return new_sol
    else:
        return sol


# mutation dans le cas où on cherche plusieurs tailles
def mutation_2(sol, centroides, numclasses, a, sl, MutProb=0.05):
    np.random.seed((666 * sl) % 10000)
    i = 1
    if numclasses == 1:
        return mutation(sol, centroides, numclasses, a, sl)
    for c in range(sol.shape[0]):
        mutate_ = np.random.choice(2, 1, p=np.array([MutProb, 1 - MutProb]))[0]
        np.random.seed((66 * sl + 30 * i + i) % 10000)
        i += 1
        if mutate_ == 0:
            set_ = set(sol.tolist()) - set([sol[c]])
            new_s = np.random.choice(centroides.shape[0], 1, replace=False)[0]
            i += 1
            iter = 0
            while new_s in set_:
                np.random.seed((666 * (sl) + i) % 10000)
                new_s = np.random.choice(centroides.shape[0], 1, replace=False)[0]

                i += 1

            sol[c] = new_s

    return sol


def selection_roulette(
    sl,
    list_,
    solutions,
    fitness,
    centroides,
    numclasses,
    init_classes,
    signature_sol,
    roulette_size=3,
):
    second_ind = 0
    first_ind = 0
    ii = 0
    iter = 0

    while True:
        first_ind = second_ind
        while first_ind == second_ind:
            np.random.seed((55 * (sl) + ii) % 10000)
            ii += 1
            first_ind = np.random.choice(
                solutions.shape[0], roulette_size, replace=False
            )

            first_ind = first_ind[np.argmin(fitness[first_ind])]
            first_ = solutions[first_ind, :]
            np.random.seed((37 * (sl) + ii) % 10000)
            second_ind = np.random.choice(
                solutions.shape[0], roulette_size, replace=False
            )

            second_ind = second_ind[np.argmin(fitness[second_ind])]
            second_ = solutions[second_ind, :]

        first_, second_ = mutation_2(
            first_, centroides, numclasses, init_classes, sl + ii
        ), mutation_2(second_, centroides, numclasses, init_classes, sl + 2 * ii + 2)
        m = merge(first_, second_, 3 * sl + ii + 4)

        if first_ind != second_ind and str(m) not in signature_sol:
            break
        elif iter == 20:
            break
        iter += 1

    list_.append(m)


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())


# optimisation local aprés chaque génération
def local_opt(
    sl,
    lock_,
    list_,
    list_2,
    df,
    sol,
    init_classes,
    centroides,
    f,
    init_classes_dic,
    computed,
    set_box,
    max_,
    columns,
    BBSS,
    pred_dic,
    poly_dic,
):
    r = [sol[0]]

    num_iter = 0
    while (
        len(set(r) & set(sol.tolist())) != 0
        or np.array(r).shape[0] != np.unique(r).shape[0]
    ):
        r = []
        np.random.seed((62 * sl + num_iter) % 10000)
        numrep = np.random.choice(max(int(sol.shape[0] / 2), 1), 1)[0] + 1
        rep = np.random.choice(sol, numrep, replace=False)

        for s in rep:
            candidates = init_classes_dic[init_classes[s]]
            np.random.seed((99 * sl + num_iter * 3 + s * 5) % 10000)
            r.append(np.random.choice(candidates, 1)[0])

        if num_iter == sol.shape[0]:
            while lock_.acquire() == False:
                None
            list_.append(sol)
            list_2.append(f)
            lock_.release()
            return
        num_iter += 1

    sol_c = copy.deepcopy(sol)
    for i in range(rep.shape[0]):
        sol_c[np.where(sol == rep[i])[0]] = r[i]

    sol_c = sol_c[np.argsort(sol_c)]
    if str(sol_c) in computed:
        f2 = computed[str(sol_c)]
    else:
        f2 = compute_fitness(
            df, sol_c, centroides, set_box, max_, columns, BBSS, pred_dic, poly_dic
        )
    # print('In  local before lock')
    while lock_.acquire() == False:
        None
    # print('In local after lock')
    if f2 < f:
        list_.append(sol_c)
        list_2.append(f2)
    else:
        list_.append(sol)
        list_2.append(f)

    lock_.release()
