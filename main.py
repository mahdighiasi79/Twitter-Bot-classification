import numpy as np
import json


def fetch_data():
    f = open("twibot20.json")
    data = json.load(f)
    return data


# core points are labeled 1
# border points are labeled 2
# noise points are labeled 0
def label_points(data, eps, k):
    m = len(data)
    labels = [0] * m

    for i in range(m):
        distances = np.linalg.norm(data - data[i], axis=1, keepdims=False)
        neighbours = distances <= eps
        neighbours = np.sum(neighbours, axis=0, keepdims=False)
        neighbours -= 1
        if neighbours >= k:
            labels[i] = 1

    return labels


def build_edges(data, eps, labels):
    edges = {}
    m = len(data)
    for i in range(m):
        if labels[i] == 1:
            edges[i] = []
            distances = np.linalg.norm(data - data[i], axis=1, keepdims=False)
            neighbours = distances <= eps
            neighbours *= labels
            neighbours[i] = 0
            for j in range(m):
                if neighbours[j] == 1:
                    edges[i].append(j)
    return edges


def core(index, edges, clusters):
    neighbours = edges[index]
    for neighbour in neighbours:
        if clusters[neighbour] == 0:
            clusters[neighbour] = clusters[index]
            core(neighbour, edges, clusters)


def cluster_cores(edges, m):
    clusters = [0] * m
    cluster_number = 0
    for key in edges:
        if clusters[key] == 0:
            cluster_number += 1
            clusters[key] = cluster_number
            core(key, edges, clusters)
    return clusters


def cluster_borders(data, eps, labels, clusters):
    m = len(data)
    for i in range(m):
        if labels[i] == 1:
            distances = np.linalg.norm(data - data[i], axis=1, keepdims=False)
            neighbours = distances <= eps
            for j in range(m):
                if neighbours[j]:
                    clusters[j] = clusters[i]
    return clusters


def dbscan(data, eps, k):
    labels = label_points(data, eps, k)
    edges = build_edges(data, eps, labels)
    clusters = cluster_cores(edges, len(data))
    clusters = cluster_borders(data, eps, labels, clusters)
    return clusters


if __name__ == '__main__':
    data = fetch_data()
    for key in data[0]["tweet"]:
        print(key)
    # print(element["profile"]["followers_count"])
    # print(element["profile"]["friends_count"])
    # print(element["profile"]["favourites_count"])
    # print(element["profile"]["listed_count"])
    # print(element["tweet"]["retweet_count"])
    # # print(profile["reply_count"])
    # print(element["tweet"]["hashtag_count"])
    # print(element["tweet"]["mentions_count"])
