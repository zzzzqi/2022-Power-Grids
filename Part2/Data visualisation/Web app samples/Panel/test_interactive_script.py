import panel as pn
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas # noqa
from hvplot import hvPlot

from sklearn import decomposition
from sklearn.cluster import KMeans, DBSCAN, k_means
import umap.umap_ as umap

# Enable Bokeh and Panel
hv.extension('bokeh')
pn.extension()

## Read the input CSV file
file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
file_input_message = "### Please upload your CNN output <br>"
metadata_df = pd.read_csv("test_predictions_01.csv", header=0, usecols=range(1, 5))
df = pd.read_csv("test_predictions_01.csv", header=0, usecols=range(6, 66))
df = df.dropna()


## Original dataframe
basic_df = pd.DataFrame(df)
basic_df_headers = list(basic_df.columns)


## Dimensionality Reduction algos
# PCA
pca = decomposition.PCA(n_components=2)
pca.fit(df)
pca_data = pca.transform(df)
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
pca_labels = ["PC 1", "PC 2"]
pca_df = pd.DataFrame(pca_data, columns=pca_labels)
pca_df_headers= list(pca_df.columns)
# UMAP
umap_reducer = umap.UMAP()
umap_reducer.fit(df)
umap_data = umap_reducer.transform(df)
umap_labels = ["UMAP 1", "UMAP 2"]
umap_df = pd.DataFrame(umap_data, columns=umap_labels)


## Column 1a: The algo-selection column
# Build the selection dropdown for Dimensionaity Reduction algos
dr_selection = pn.widgets.Select(
    options=["Nil", "PCA", "UMAP"]
)
# Build the selection dropdown for Clustering algos
clustering_selection = pn.widgets.Select(
    options=["Nil", "K-Means", "DBSCAN"]
)
# Build the inter-active algo-selection message
@pn.depends(dr_selection.param.value, clustering_selection.param.value)
def algo_message(dr_value, clustering_value):
    return pn.pane.Markdown(
        "*The selected algos: <br>*" +
        "*Dimensionality reduction: " + str(dr_value) + "<br>*" +
        "*Clustering: " + str(clustering_value) + "<br>*"
    )
# Build the WidgetBox for algos selection
algo_column = pn.WidgetBox(
    pn.pane.Markdown("### Dimensionality reduction algos: <br>"),
    dr_selection,
    pn.pane.Markdown("### Clustering algos: <br>"),
    clustering_selection,
)


## Column 1b: The plot-configuration column
# Basic selection options
basic_df_x_axis_selection = pn.widgets.Select(
    name="X-axis: ",
    options=basic_df_headers
)
basic_df_y_axis_selection = pn.widgets.Select(
    name="Y-axis: ",
    options=basic_df_headers,
    value=basic_df_headers[1]
)
# PCA selection options
pca_df_x_axis_selection = pn.widgets.Select(
    name="X-axis: ",
    options=pca_df_headers
)
pca_df_y_axis_selection = pn.widgets.Select(
    name="Y-axis: ",
    options=pca_df_headers,
    value=pca_df_headers[1]
)
# UMAP selection options
umap_df_x_axis_selection = pn.widgets.Select(
    name="X-axis: ",
    options=umap_labels
)
umap_df_y_axis_selection = pn.widgets.Select(
    name="Y-axis: ",
    options=umap_labels,
    value=umap_labels[1]
)
# K-Means clustering selection options
k_means_n_clusters_selection = pn.widgets.IntSlider(
    value=5,
    start=1,
    end=10,
    step=1,
    name="Number of clusters"
)
# DBSCAN selection options
dbscan_max_distance_selection = pn.widgets.FloatSlider(
    value=0.5,
    start=1.0,
    end=10.0,
    step=0.5,
    name="Max distance between samples"
)
dbscan_n_samples_selection = pn.widgets.IntSlider(
    value=5,
    start=0,
    end=50,
    step=5,
    name="Number of samples in a neighbourhood"
)
# Build the WidgetBox for plot configuration
@pn.depends(dr_selection.param.value, clustering_selection.param.value)
def plot_configuration(dr_value, clustering_value):
    if dr_value == "Nil" and clustering_value == "Nil":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            basic_df_x_axis_selection,
            basic_df_y_axis_selection,
        )
    elif dr_value == "PCA" and clustering_value == "Nil":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            pca_df_x_axis_selection,
            pca_df_y_axis_selection,
        )
    elif dr_value == "UMAP" and clustering_value == "Nil":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            umap_df_x_axis_selection,
            umap_df_y_axis_selection,
        )
    elif dr_value == "Nil" and clustering_value == "K-Means":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            basic_df_x_axis_selection,
            basic_df_y_axis_selection,
            k_means_n_clusters_selection
        )
    elif dr_value == "PCA" and clustering_value == "K-Means":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            pca_df_x_axis_selection,
            pca_df_y_axis_selection,
            k_means_n_clusters_selection
        )
    elif dr_value == "UMAP" and clustering_value == "K-Means":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            umap_df_x_axis_selection,
            umap_df_y_axis_selection,
            k_means_n_clusters_selection
        )
    elif dr_value == "Nil" and clustering_value == "DBSCAN":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            basic_df_x_axis_selection,
            basic_df_y_axis_selection,
            dbscan_max_distance_selection,
            dbscan_n_samples_selection
        )
    elif dr_value == "PCA" and clustering_value == "DBSCAN":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            pca_df_x_axis_selection,
            pca_df_y_axis_selection,
            dbscan_max_distance_selection,
            dbscan_n_samples_selection
        )
    elif dr_value == "UMAP" and clustering_value == "DBSCAN":
        return pn.WidgetBox(
            pn.pane.Markdown("### Data-exploration pane options: "),
            umap_df_x_axis_selection,
            umap_df_y_axis_selection,
            dbscan_max_distance_selection,
            dbscan_n_samples_selection
        )


## Column 2: The data-exploration pane
# Build the inter-active data-exploration pane
@pn.depends(dr_selection.param.value, clustering_selection.param.value,
            basic_df_x_axis_selection.param.value, basic_df_y_axis_selection.param.value,
            pca_df_x_axis_selection.param.value, pca_df_y_axis_selection.param.value,
            umap_df_x_axis_selection.param.value, umap_df_y_axis_selection.param.value,
            k_means_n_clusters_selection.param.value,
            dbscan_max_distance_selection.param.value, dbscan_n_samples_selection.param.value)
def data_exploration(dr_value, clustering_value, 
                    basic_x_value, basic_y_value, 
                    pca_x_value, pca_y_value,
                    umap_x_value, umap_y_value,
                    k_means_n_clusters,
                    dbscan_max_distance_value, dbscan_n_samples_value):
    if dr_value == "Nil" and clustering_value == "Nil":
        return basic_df.hvplot.scatter(
            x=basic_x_value,
            y=basic_y_value,
            height=600,
            width=700,
            title="Data-exploration pane"
        )
    elif dr_value == "PCA" and clustering_value == "Nil":
        return pca_df.hvplot.scatter(
            x=pca_x_value,
            y=pca_y_value,
            height=600,
            width=700,
            title="Data-exploration pane"
        )
    elif dr_value == "UMAP" and clustering_value == "Nil":
        return umap_df.hvplot.scatter(
            x=umap_x_value,
            y=umap_y_value,
            height=600,
            width=700,
            title="Data-exploration pane"
        )
    elif dr_value == "Nil" and clustering_value == "K-Means":
        basic_kmeans = KMeans(n_clusters=k_means_n_clusters)
        y_pred = basic_kmeans.fit_predict(basic_df)

        plot = basic_df.hvplot.scatter(
            x=basic_x_value,
            y=basic_y_value,
            c=y_pred,
            cmap="rainbow",
            height=600,
            width=700,
            title="Data-exploration pane"
        )
        return plot
    elif dr_value == "PCA" and clustering_value == "K-Means":
        pca_kmeans = KMeans(n_clusters=k_means_n_clusters)
        y_pred = pca_kmeans.fit_predict(pca_df)

        plot = pca_df.hvplot.scatter(
            x=pca_x_value,
            y=pca_y_value,
            c=y_pred,
            cmap="rainbow",
            height=600,
            width=700,
            title="Data-exploration pane"
        )
        return plot
    elif dr_value == "UMAP" and clustering_value == "K-Means":
        umap_kmeans = KMeans(n_clusters=k_means_n_clusters)
        y_pred = umap_kmeans.fit_predict(umap_df)
        
        plot = umap_df.hvplot.scatter(
            x=umap_x_value,
            y=umap_y_value,
            c=y_pred,
            cmap="rainbow",
            height=600,
            width=700,
            title="Data-exploration pane"
        )
        return plot
    elif dr_value == "Nil" and clustering_value == "DBSCAN":
        basic_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
        y_pred = basic_dbscan.fit_predict(df)
        labels = basic_dbscan.labels_

        plot = basic_df.hvplot.scatter(
            x=basic_x_value,
            y=basic_y_value,
            c=y_pred,
            cmap="bmw",
            height=600,
            width=700,
            title="Data-exploration pane"
        )
        return plot
    elif dr_value == "PCA" and clustering_value == "DBSCAN":
        pca_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
        y_pred = pca_dbscan.fit_predict(pca_df)
        labels = pca_dbscan.labels_

        plot = pca_df.hvplot.scatter(
            x=pca_x_value,
            y=pca_y_value,
            c=y_pred,
            cmap="bmw",
            height=600,
            width=700,
            title="Data-exploration pane"
        )
        return plot
    elif dr_value == "UMAP" and clustering_value == "DBSCAN":
        umap_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
        y_pred = umap_dbscan.fit_predict(umap_df)
        labels = umap_dbscan.labels_
        
        plot = umap_df.hvplot.scatter(
            x=umap_x_value,
            y=umap_y_value,
            c=y_pred,
            cmap="bmw",
            height=600,
            width=700,
            title="Data-exploration pane"
        )
        return plot


## Build the layout of the dashboard
# Set the template used in the dashboard
material = pn.template.MaterialTemplate(title='Smart Power Grids Dashboard')
# Set the dashboard's header contents
header = pn.Row(
    pn.layout.HSpacer(),
    pn.Spacer(width=100)
)
# Set the dashboard's instructions
instructions = """
        ## Welcome to the Smart Power Grids Dashboard <br>
        ### Instructions <br>
        <ol>
        <li> Upload the CNN output to the dashboard <br>
        <li> Choose the Dimensionality Reduction algo and the corresponding variables to plot the data-exploration pane <br>
        <li> Choose the Clustering algo to perform unspervised grouping <br>
        <li> Select a dot from the plot. Open a window with the followings: <br>
        <ul>
        <li> This event's 6 waveforms (3 voltages, 3 currents)
        <li> Similar events to this
        <li> The 6 waveforms of one of the similar events
        </ul>
        <li> Perform grouping and labelling on the plot <br>
        <li> Export the data of the grouped events in CSV <br>
        </ol>
"""
message = pn.Column( 
    instructions, 
    sizing_mode="stretch_width"
)

# Add the dashboard's header
material.header.append(header)
# Add the dashboard's sidebar with the file upload widget
material.sidebar.append(message)
material.sidebar.append(file_input_message)
material.sidebar.append(file_input)
# Add the dashboard's data-exploration panes
material.main.append(
    pn.Row(
        pn.Column(
            algo_column,
            plot_configuration
        ),
        pn.Column(
            data_exploration
        )
    )
)

material.servable()
# material.save('test_interactive_script.html', resources=INLINE)
