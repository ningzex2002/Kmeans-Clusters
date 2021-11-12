# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:58:34 2021

@author: ningz
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import sklearn as sk


st.title("Kmeans Clusters")
max_iter = st.slider("Choose the number of iterations",1)

X, _ = sk.datasets.make_blobs(n_samples=1000, centers=5, n_features=2, random_state = 1)
df = pd.DataFrame(X, columns = list("ab"))
starting_points = np.array([[0,0],[-2,0],[-4,0],[0,2],[0,4]])
kmeans = sk.cluster.KMeans(n_clusters = 5, max_iter=max_iter, init=starting_points, n_init = 1)
kmeans.fit(X);
df["c"] = kmeans.predict(X)
chart1 = alt.Chart(df).mark_circle().encode(
    x = "a",
    y = "b",
    color = "c:N"
)

df_centers = pd.DataFrame(kmeans.cluster_centers_, columns = list("ab"))

chart_centers = alt.Chart(df_centers).mark_point().encode(
    x = "a",
    y = "b",
    color = alt.value("black"),
    shape = alt.value("diamond"),
)

chart1 + chart_centers

