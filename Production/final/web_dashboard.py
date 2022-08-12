## Name: web_dashboard.py
## Date: 23 Aug 2022
## By: Group F Toumetis - MSc CS Final Project - University of Bristol

import io
import os
import panel as pn
import numpy as np
import pandas as pd
import altair as alt
import holoviews as hv
import hvplot.pandas  # noqa
from math import sqrt
from hvplot import hvPlot
from bokeh.resources import INLINE

from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import umap.umap_ as umap

# Enable Bokeh and Panel
hv.extension('bokeh')
pn.extension('vega', 'tabulator')

## ========================================================
## Build the layout of the dashboard
# Set the dashboard template
app = pn.template.MaterialTemplate(title='Smart Power Grids Dashboard')
# Set the dashboard's instructions
instructions = """
    #### This dashboard facilitates the discovery of similar power signal events. <br>
    To discover the events: <br>
    <ul>
    <li> Upload the CNN output file <br>
    <li> Choose a dimensionality reduction algorithm to adjust the number of features considered <br>
    <li> Choose a clustering algorithm to identify similar events <br>
    <li> Select the axis options of the Data-exploration pane to adjust how the pane is plotted <br>
    <li> Select a specific point on the pane <br>
    <li> Switch to the Power signal event tab to review the corresponding event details, including: <br>
    <ol>
    <li> Event data
    <li> Event waveforms
    <li> A summary of similar events
    </ol>
    <li> Change the similar events options to adjust how the similar events are found <br>
    <li> Choose the events from the summary, click the "Display selected events" button, 
    and display the selected events to the side <br>
    <li> Choose a group of events from the summary, provide a group name, 
    and export the corresponding event data to a CSV file <br>
    </ul>
    """
file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
file_input_message = "#### Upload the CNN output file: <br>"
# Set the dashboard's sidebar
app.sidebar.append(instructions)
app.sidebar.append(
    pn.Spacer(
        width=330,
        height=1,
        background="lightgrey"
    )
)
app.sidebar.append(file_input_message)
app.sidebar.append(file_input)


## ========================================================
## Read the input CSV file and set it for the dynamic environment
@pn.depends(file_input)
def dynamic_env(read_file):
    """
    This method reads the CNN output file as the input 
    and uses it for the dynamic environment of the dashboard.

    :param read_file: the CNN output file
    :return: the dynamic environment of the dashboard
    """
    ## ========================================================
    ## Load the CSV file
    # Read the input CSV file once provided
    # Otherwise generate dummy data for the dashboard
    if read_file is None:
        metadata_columns = ["metadata_dummy_axis" + str(i) for i in range(0, 4)]
        metadata_columns[0] = "event_id"
        metadata_df = pd.DataFrame(np.random.uniform(0, 1, size=(100, 4)), columns=metadata_columns)
        df = pd.DataFrame(np.random.uniform(0, 1, size=(100, 60)),
                          columns=["dummy_axis" + str(i) for i in range(0, 60)])
    else:
        metadata_df = pd.read_csv(io.BytesIO(file_input.value), header=0, usecols=range(0, 4))
        df = pd.read_csv(io.BytesIO(file_input.value), header=0, usecols=range(4, 64))
    metadata_df = metadata_df.dropna()
    df = df.dropna()

    ## ========================================================
    ## Define settings for the widgetboxes at the sidebar
    widgetbox_width = 330

    ## ========================================================
    ## Define the basic dataframe
    basic_df = pd.DataFrame(df)
    basic_df_headers = list(basic_df.columns)

    ## ========================================================
    ## Define the axes of Dimensionality Reduction algos
    # PCA
    pca_labels = ["PC 1", "PC 2"]
    # UMAP
    umap_labels = ["UMAP 1", "UMAP 2"]
    # t-SNE
    tsne_labels = ["t-SNE 1", "t-SNE 2"]

    ## ========================================================
    ## Column 1a: The algo-selection column
    ## Build the selection options for the algos
    # Build the selection dropdown for Dimensionaity Reduction algos
    dr_selection = pn.widgets.Select(
        options=["Nil", "PCA", "UMAP", "t-SNE"]
    )
    # Build the selection dropdown for Clustering algos
    clustering_selection = pn.widgets.Select(
        options=["Nil", "K-Means", "DBSCAN", "Agglomerative Clustering"]
    )
    # Build the WidgetBox for algos selection
    algo_column = pn.WidgetBox(
        pn.pane.Markdown("#### Dimensionality reduction algos: "),
        dr_selection,
        pn.pane.Markdown("#### Clustering algos: "),
        clustering_selection,
        pn.pane.Markdown(""),
        width=widgetbox_width
    )

    ## ========================================================
    ## Column 1b: The plot-configuration column
    ## Build the plot configuration options
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
        options=pca_labels
    )
    pca_df_y_axis_selection = pn.widgets.Select(
        name="Y-axis: ",
        options=pca_labels,
        value=pca_labels[1]
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
    # t-SNE selection options
    tsne_df_x_axis_selection = pn.widgets.Select(
        name="X-axis: ",
        options=tsne_labels
    )
    tsne_df_y_axis_selection = pn.widgets.Select(
        name="Y-axis: ",
        options=tsne_labels,
        value=tsne_labels[1]
    )

    # PCA parameters
    pca_whiten = pn.widgets.Select(
        name="whiten: ",
        options={
            'False': False,
            'True': True
        }
    )
    pca_svd_solver = pn.widgets.Select(
        name="svd_solver: ",
        options={
            'auto': 'auto',
            'full': 'full',
            'arpack': 'arpack',
            'randomized': 'randomized'
        }
    )
    # UMAP parameters
    umap_n_neighbors = pn.widgets.IntSlider(
        name="n_neighbors: ",
        value=15,
        start=0,
        end=200,
        step=5
    )
    umap_min_dist = pn.widgets.FloatSlider(
        name="min_dist: ",
        value=0.1,
        start=0,
        end=0.99,
        step=0.01
    )
    # t-SNE parameters
    tsne_perplexity = pn.widgets.FloatSlider(
        name="perplexity: ",
        value=30.0,
        start=5,
        end=50,
        step=1
    )
    tsne_early_exaggeration = pn.widgets.FloatSlider(
        name="early_exaggeration: ",
        value=12.0,
        start=2,
        end=100,
        step=1
    )
    tsne_learning_rate = pn.widgets.FloatSlider(
        name="learning_rate: ",
        value=200.0,
        start=10.0,
        end=1000.0,
        step=10
    )
    # K-Means parameters
    k_means_n_clusters_selection = pn.widgets.IntSlider(
        name="n_clusters: ",
        value=5,
        start=1,
        end=10,
        step=1,
    )
    # DBSCAN parameters
    dbscan_max_distance_selection = pn.widgets.FloatSlider(
        name="eps: ",
        value=0.5,
        start=0.1,
        end=10.0,
        step=0.1,
    )
    dbscan_n_samples_selection = pn.widgets.IntSlider(
        name="min_samples: ",
        value=10,
        start=5,
        end=50,
        step=5,
    )
    # Agglomerative Clustering parameters
    agg_clustering_n_clusters_selection = pn.widgets.IntSlider(
        name="n_clusters: ",
        value=5,
        start=1,
        end=10,
        step=1,
    )
    agg_clustering_linkage_selection = pn.widgets.Select(
        name="linkage: ",
        options={
            'ward': 'ward',
            'complete': 'complete',
            'average': 'average',
            'single': 'single'
        }
    )

    # Build the dynamic widgetbox for plot configuration
    # Dynamic in the sense that it depends on the
    # Dimensionality Reduction and Clustering algos chosen by the user
    @pn.depends(dr_selection.param.value, clustering_selection.param.value)
    def plot_configuration(dr_value, clustering_value):
        """
        This method builds the dynamic widgetbox for plot configuration.
        It depends on the selected Dimensionality Reduction and Clustering algos.

        :param dr_value: the selected Dimensionality Reduction algo
        :param clustering_value: the selected Clustering algo
        :return: the dynamic widgetbox for plot configuration, listing 
            the axis options for the scatter-plot, and 
            the parameter options for the selected algos
        """
        if dr_value == "Nil" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "PCA" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                pn.pane.Markdown("#### PCA parameters: "),
                pca_whiten,
                pca_svd_solver,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "UMAP" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                pn.pane.Markdown("#### UMAP parameters: "),
                umap_n_neighbors,
                umap_min_dist,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "Nil" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                pn.pane.Markdown("#### K-Means parameters: "),
                k_means_n_clusters_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "PCA" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                pn.pane.Markdown("#### PCA parameters: "),
                pca_whiten,
                pca_svd_solver,
                pn.pane.Markdown("#### K-Means parameters: "),
                k_means_n_clusters_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "UMAP" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                pn.pane.Markdown("#### UMAP parameters: "),
                umap_n_neighbors,
                umap_min_dist,
                pn.pane.Markdown("#### K-Means parameters: "),
                k_means_n_clusters_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "Nil" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                pn.pane.Markdown("#### DBSCAN parameters: "),
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "PCA" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                pn.pane.Markdown("#### PCA parameters: "),
                pca_whiten,
                pca_svd_solver,
                pn.pane.Markdown("#### DBSCAN parameters: "),
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "UMAP" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                pn.pane.Markdown("#### UMAP parameters: "),
                umap_n_neighbors,
                umap_min_dist,
                pn.pane.Markdown("#### DBSCAN parameters: "),
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "t-SNE" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                tsne_df_x_axis_selection,
                tsne_df_y_axis_selection,
                pn.pane.Markdown("#### t-SNE parameters: "),
                tsne_perplexity,
                tsne_early_exaggeration,
                tsne_learning_rate,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "t-SNE" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                tsne_df_x_axis_selection,
                tsne_df_y_axis_selection,
                pn.pane.Markdown("#### t-SNE parameters: "),
                tsne_perplexity,
                tsne_early_exaggeration,
                tsne_learning_rate,
                pn.pane.Markdown("#### K-Means parameters: "),
                k_means_n_clusters_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "t-SNE" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                tsne_df_x_axis_selection,
                tsne_df_y_axis_selection,
                pn.pane.Markdown("#### t-SNE parameters: "),
                tsne_perplexity,
                tsne_early_exaggeration,
                tsne_learning_rate,
                pn.pane.Markdown("#### DBSCAN parameters: "),
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "Nil" and clustering_value == "Agglomerative Clustering":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                pn.pane.Markdown("#### Agglomerative Clustering parameters: "),
                agg_clustering_n_clusters_selection,
                agg_clustering_linkage_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "PCA" and clustering_value == "Agglomerative Clustering":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                pn.pane.Markdown("#### PCA parameters: "),
                pca_whiten,
                pca_svd_solver,
                pn.pane.Markdown("#### Agglomerative Clustering parameters: "),
                agg_clustering_n_clusters_selection,
                agg_clustering_linkage_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "UMAP" and clustering_value == "Agglomerative Clustering":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                pn.pane.Markdown("#### UMAP parameters: "),
                umap_n_neighbors,
                umap_min_dist,
                pn.pane.Markdown("#### Agglomerative Clustering parameters: "),
                agg_clustering_n_clusters_selection,
                agg_clustering_linkage_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "t-SNE" and clustering_value == "Agglomerative Clustering":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                tsne_df_x_axis_selection,
                tsne_df_y_axis_selection,
                pn.pane.Markdown("#### t-SNE parameters: "),
                tsne_perplexity,
                tsne_early_exaggeration,
                tsne_learning_rate,
                pn.pane.Markdown("#### Agglomerative Clustering parameters: "),
                agg_clustering_n_clusters_selection,
                agg_clustering_linkage_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )

    ## ========================================================
    ## Column 1c: The similar events configuration column
    # Similar events options for NO DR nor NO Clustering algos
    basic_similar_events_x_value_selection = pn.widgets.FloatSlider(
        name="Selected event's x-axis prediction score +/ -",
        value=0.05,
        start=0.00,
        end=1.00,
        step=0.05,
    )
    basic_similar_events_y_value_selection = pn.widgets.FloatSlider(
        name="Selected event's y-axis prediction score +/ -",
        value=0.05,
        start=0.00,
        end=1.00,
        step=0.05,
    )
    # Similar events options for YES DR and NO Clustering algos
    dr_similar_events_x_value_selection = pn.widgets.FloatSlider(
        name="Selected event's x-value percentage +/ -",
        value=0.15,
        start=0.00,
        end=1.00,
        step=0.05,
    )
    dr_similar_events_y_value_selection = pn.widgets.FloatSlider(
        name="Selected event's y-value percentage +/ -",
        value=0.15,
        start=0.00,
        end=1.00,
        step=0.05,
    )

    # Build the dynamic similar events configuration widgetbox
    # Dynamic in the sense that it depends on the
    # Dimensionality Reduction and Clustering algos chosen by the user
    @pn.depends(dr_selection.param.value, clustering_selection.param.value)
    def similar_events_configuration(dr_value, clustering_value):
        """
        This method builds the dynamic widgetbox for similar events configuration.
        It depends on the selected Dimensionality Reduction and Clustering algos.

        :param dr_value: the selected Dimensionality Reduction algo
        :param clustering_value: the selected Clustering algo
        :return: the dynamic widgetbox for configuring similar events, listing 
            the options for changing how similar events are identified
        """
        if dr_value == "Nil" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Similar events options: "),
                basic_similar_events_x_value_selection,
                basic_similar_events_y_value_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "PCA" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Similar events options: "),
                dr_similar_events_x_value_selection,
                dr_similar_events_y_value_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "UMAP" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Similar events options: "),
                dr_similar_events_x_value_selection,
                dr_similar_events_y_value_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        elif dr_value == "t-SNE" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Similar events options: "),
                dr_similar_events_x_value_selection,
                dr_similar_events_y_value_selection,
                pn.pane.Markdown(""),
                width=widgetbox_width
            )
        else:
            return pn.WidgetBox(
                pn.pane.Markdown("#### Similar events options: "),
                pn.pane.Markdown("Similar events are selected based on their cluster values."),
                pn.pane.Markdown(""),
                width=widgetbox_width
            )

    ## ========================================================
    ## Column 2: The data-exploration pane
    # Define the general settings for the charts
    data_exploration_pane_height = 630
    data_exploration_pane_width = 700
    waveform_height = 400
    waveform_width = 975
    waveform_fontsize = 0.7

    # Build a helper function to identify the top PQD predictions for the waveforms
    # Return a dictionary with the waveform names as keys, and their top PQD types and scores as values
    def identify_top_predictions(df, event_id):
        """
        This method identifies the top PQD predictions of the signal waveforms.

        :param df: the dataframe of the selected event
        :param event_id: the id of the selected event
        :return: the dictionary where waveform names are keys, 
            and the arrays of their top PQD types and prediction scores 
            are the values
        """
        df.set_index("event_id", inplace=True)
        waveform_names = [
            "Vab", "Vbc", "Vca", "Ia", "Ib", "Ic"
        ]
        pqd_names = [
            "flickers", "harmonics", "interruptions", "interruptions_harmonics", "osc_transients",
            "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"
        ]
        waveform_predictions_dict = dict()

        for i in range(len(waveform_names)):
            target_waveform = waveform_names[i]
            top_pqd_name = ""
            top_pqd_score = -1
            for j in range(len(pqd_names)):
                target_column_name = target_waveform.lower() + "_" + pqd_names[j]
                target_column_value = df.loc[event_id, target_column_name]

                if target_column_value > top_pqd_score:
                    top_pqd_name = pqd_names[j]
                    top_pqd_score = target_column_value
            waveform_predictions_dict[target_waveform] = [top_pqd_name, top_pqd_score]

        return waveform_predictions_dict

    # Build the dynamic event page
    # Dynamic in the sense that it depends on 
    # the event selection on the data-exploraton pane
    def build_event_page(
            selection,
            selected_df,
            x_axis,
            y_axis,
            clusters):
        """
        This method builds the dynamic event page.
        It depends on the selected event on the data-exploration pane.

        :param selection: the selected event on the data-exploration pane
        :param selected_df: the dataframe of the selected event and its similar events modified by the selected axes and algos 
        :param x_axis: the selected x-axis
        :param y_axis: the selected y-axis
        :param clusters: the selected Clustering algo
        :return: the dynamic event page that shows the event data, and
            the event waveforms of the selected event
        """

        # Select the event data CSV file
        selected_event_df = selected_df.iloc[selection[0] - 1]
        selected_event_csv_filename = os.getcwd() + os.sep + "event_data" + os.sep + \
                                      selected_event_df["input_event_csv_filename"] + ".csv"
        selected_event_details_df = pd.read_csv(selected_event_csv_filename, header=0)

        # Extract the event data
        selected_event_id = selected_event_df["event_id"]
        selected_event_x_value = selected_event_df[x_axis]
        selected_event_y_value = selected_event_df[y_axis]
        selected_event_predictions_dict = identify_top_predictions(
            pd.concat([metadata_df, df], axis=1), selected_event_id)

        # Build a Row for the event data
        event_data = pn.Row(
            pn.Column(
                f"""
                ** _Event metadata:_ **
                <ul>
                <li> event_id: {str(selected_event_id)}
                <li> start_time: {str(selected_event_df["start_time"])}
                <li> asset_name: {str(selected_event_df["asset_name"])}
                </ul>
                """,
                width=300
            ),
            pn.Spacer(
                background="lightgrey",
                width=1,
                height=110
            ),
            pn.Column(
                f"""
                ** _Axis values:_ **
                <ul>
                <li> {str(x_axis)}: {"%.5f" % selected_event_x_value}
                <li> {str(y_axis)}: {"%.5f" % selected_event_y_value}
                </ul>
                """,
                width=300
            )
        )
        if clusters != None:
            event_data.append(
                pn.Spacer(
                    background="lightgrey",
                    width=1,
                    height=110
                )
            )
            event_data.append(
                pn.Column(
                    f"""
                    ** _Cluster value:_ **
                    <ul>
                    <li> cluster: {str(selected_event_df["cluster"])}
                    </ul>
                    """,
                )
            )
        event_waveform_predictons = pn.Row(
            pn.Column(
                f"""
                ** _Top PQD predictions for event's voltages:_ **
                <ul>
                <li> Vab: {selected_event_predictions_dict["Vab"][0]} 
                    ({"%.5f" % selected_event_predictions_dict["Vab"][1]})
                <li> Vbc: {selected_event_predictions_dict["Vbc"][0]} 
                    ({"%.5f" % selected_event_predictions_dict["Vbc"][1]})
                <li> Vca: {selected_event_predictions_dict["Vca"][0]} 
                    ({"%.5f" % selected_event_predictions_dict["Vca"][1]})
                </ul>
                """,
                width=300
            ),
            pn.Spacer(
                background="lightgrey",
                width=1,
                height=110
            ),
            pn.Column(
                f"""
                ** _Top PQD predictions for event's currents:_ **
                <ul>
                <li> Ia: {selected_event_predictions_dict["Ia"][0]} 
                    ({"%.5f" % selected_event_predictions_dict["Ia"][1]})
                <li> Ib: {selected_event_predictions_dict["Ib"][0]} 
                    ({"%.5f" % selected_event_predictions_dict["Ib"][1]})
                <li> Ic: {selected_event_predictions_dict["Ic"][0]} 
                    ({"%.5f" % selected_event_predictions_dict["Ic"][1]})
                </ul>
                """,
                width=300
            )
        )

        # Plot the event waveforms
        voltages_waveforms = selected_event_details_df.hvplot.line(
            x='timestamps (in ms)',
            y=['Vab', 'Vbc', 'Vca'],
            legend=True,
            responsive=False,
            height=waveform_height,
            width=waveform_width,
            title="Voltages"
        ).opts(
            fontscale=waveform_fontsize,
        )
        currents_waveforms = selected_event_details_df.hvplot.line(
            x='timestamps (in ms)',
            y=['Ia', 'Ib', 'Ic'],
            legend=True,
            responsive=False,
            height=waveform_height,
            width=waveform_width,
            title="Currents"
        ).opts(
            fontscale=waveform_fontsize
        )

        # Build the entire event page with event data and waveforms
        event_page = pn.Row(
            pn.Column(
                "#### Selected event:",
                # Event page element - metadata
                pn.WidgetBox(
                    "#### Event data:",
                    event_data,
                    pn.Spacer(
                        background="lightgrey",
                        width=975,
                        height=1
                    ),
                    event_waveform_predictons,
                    width=1000
                ),
                # Event page element - waveforms
                pn.WidgetBox(
                    "#### Event waveforms:",
                    voltages_waveforms,
                    currents_waveforms,
                    width=1000
                )
            )
        )
        return event_page

    # Build the dynamic similar event page
    # Dynamic in the sense that it depends on 
    # the event selection on the data-exploraton pane,
    # and the similar events parameters selected
    def build_similar_event_page(
            selection,
            selected_df,
            x_axis,
            y_axis,
            clusters,
            similar_events_x_parameter,
            similar_events_y_parameter):
        """
        This method builds the dynamic page of similar events.
        It depends on the selected event on the data-exploration pane, and 
        the parameters chosen for configuring the similar event selection.

        :param selection: the selected event on the data-exploration pane
        :param selected_df: the dataframe of the selected event and its similar events modified by the selected axes and algos 
        :param x_axis: the selected x-axis
        :param y_axis: the selected y-axis
        :param clusters: the selected Clustering algo
        :param similar_events_x_parameter: the selected x-value for configuring the similar events
        :param similar_events_y_parameter: the selected y-value for configuring the similar events
        :return: the dynamic page of similar events that shows the Tabulator object
            of the similar events identified
        """

        # Extract the event data
        selected_event_df = selected_df.iloc[selection[0] - 1]
        selected_event_x_value = selected_event_df[x_axis]
        selected_event_y_value = selected_event_df[y_axis]

        # Identify the similar events
        # For no clustering algos
        if (clusters == None):
            # For no Dimensionality Reduction algos
            if (x_axis != "PC 1" and x_axis != "PC 2" and x_axis != "UMAP 1" and x_axis != "UMAP 2"):
                similar_events_df = selected_df.loc[(
                        (selected_df[x_axis] <= selected_event_x_value + similar_events_x_parameter) &
                        (selected_df[x_axis] >= selected_event_x_value - similar_events_x_parameter) &
                        (selected_df[y_axis] <= selected_event_y_value + similar_events_y_parameter) &
                        (selected_df[y_axis] >= selected_event_y_value - similar_events_y_parameter))]
            # For yes Dimensionality Reduction algos
            else:
                if selected_event_x_value >= 0 and selected_event_y_value >= 0:
                    similar_events_df = selected_df.loc[(
                            (selected_df[x_axis] <= selected_event_x_value * (1 + similar_events_x_parameter)) &
                            (selected_df[x_axis] >= selected_event_x_value * (1 - similar_events_x_parameter)) &
                            (selected_df[y_axis] <= selected_event_y_value * (1 + similar_events_y_parameter)) &
                            (selected_df[y_axis] >= selected_event_y_value * (1 - similar_events_y_parameter)))]
                elif selected_event_x_value < 0 and selected_event_y_value >= 0:
                    similar_events_df = selected_df.loc[(
                            (selected_df[x_axis] >= selected_event_x_value * (1 + similar_events_x_parameter)) &
                            (selected_df[x_axis] <= selected_event_x_value * (1 - similar_events_x_parameter)) &
                            (selected_df[y_axis] <= selected_event_y_value * (1 + similar_events_y_parameter)) &
                            (selected_df[y_axis] >= selected_event_y_value * (1 - similar_events_y_parameter)))]
                elif selected_event_x_value >= 0 and selected_event_y_value < 0:
                    similar_events_df = selected_df.loc[(
                            (selected_df[x_axis] <= selected_event_x_value * (1 + similar_events_x_parameter)) &
                            (selected_df[x_axis] >= selected_event_x_value * (1 - similar_events_x_parameter)) &
                            (selected_df[y_axis] >= selected_event_y_value * (1 + similar_events_y_parameter)) &
                            (selected_df[y_axis] <= selected_event_y_value * (1 - similar_events_y_parameter)))]
                else:
                    similar_events_df = selected_df.loc[(
                            (selected_df[x_axis] >= selected_event_x_value * (1 + similar_events_x_parameter)) &
                            (selected_df[x_axis] <= selected_event_x_value * (1 - similar_events_x_parameter)) &
                            (selected_df[y_axis] >= selected_event_y_value * (1 + similar_events_y_parameter)) &
                            (selected_df[y_axis] <= selected_event_y_value * (1 - similar_events_y_parameter)))]
        # For clustering algos
        else:
            # Add cluster value to the similar events dataframe
            selected_event_cluster = selected_event_df["cluster"]
            similar_events_df = selected_df.loc[(selected_df["cluster"] == selected_event_cluster)]
            # Calculate the Euclidean distance of the similar events in the same cluster to the selected event
            # Then add it to the similar events dataframe
            euclidean_distances = []
            for i in range(len(similar_events_df)):
                x_distance = similar_events_df.iloc[i][x_axis] - selected_event_x_value
                y_distance = similar_events_df.iloc[i][y_axis] - selected_event_y_value
                euclidean_distances.append(sqrt(x_distance ** 2 + y_distance ** 2))
            similar_events_df.insert(
                len(similar_events_df.columns) - 1,
                "Euclidean distance",
                euclidean_distances
            )
        similar_events_df.set_index("event_id", inplace=True)

        # Build similar events summary as a Tabulator
        similar_events_tabs = pn.widgets.Tabulator(
            similar_events_df,
            header_filters=True,
            selectable="checkboxes",
            layout="fit_data_fill",
            disabled=True,
            page_size=10000,
            width=950
        )

        # Build a download widget to export a CSV file of the selected events
        file_name_input_widgets = pn.widgets.TextInput(
            placeholder='filename.csv',
            width=200,
            max_height=30
        )
        group_name_widgets = pn.widgets.TextInput(
            placeholder='groupname',
            width=200,
            max_height=30
        )
        def _download_callback():
            """
            The method is a callback activated by the download button.

            :return: the CSV file of the summary of the selected events
            """
            sio = io.StringIO()
            similar_events_selection = similar_events_tabs.selection
            similar_events_df_copy = similar_events_df.reset_index()
            
            selected_similar_events_df = \
                pd.DataFrame(columns=similar_events_df_copy.columns)
            for i in range(len(similar_events_selection)):
                target_index = similar_events_selection[i]
                selected_similar_events_df.loc[i] = \
                (similar_events_df_copy.iloc[target_index])
            
            selected_similar_events_df.insert(
                len(selected_similar_events_df.columns),
                'group_name',
                group_name_widgets.value)
            
            selected_similar_events_df.to_csv(sio, index=False)
            sio.seek(0)
            return sio
        download_widget = pn.widgets.FileDownload(
            filename="filename.csv",
            callback=_download_callback,
            button_type="success",
            width=200,
            height=20
        )

        file_name_input_widgets.link(download_widget, value='filename')

        # Build a Button to display the selected similar events
        display_similar_events_button = pn.widgets.Button(
            name="Display selected events",
            button_type="success",
            width=200,
            margin=(0, 0, 0, 0)
        )

        # Build a Row for the entire similar events page
        similar_event_page = pn.Row(
            pn.Column(
                # Event page element - similar events summary
                "#### Similar events:",
                pn.WidgetBox(
                    pn.Row(
                        pn.Column(
                            f"""
                            ** _Identifying similar events:_ **
                            <ul>
                            <li> A total of {str(len(similar_events_df.index) - 1)} similar events are identified.
                            <li> To display the selected events from the summary below:
                            </ul>
                            _Note: Displaying more events leads to slower performance._
                            """,
                            width=370
                        ),
                        pn.Column(
                            "<br>",
                            display_similar_events_button,
                        )
                    ),
                    pn.Spacer(
                        width=950,
                        height=1,
                        background="lightgrey"
                    ),
                    pn.Row(
                        pn.Column(
                            """
                            ** _Labelling a group of events:_ **
                            <ul>
                            <li> To enter the group name for the selected events: <br>
                            <li> To enter the filename for the export data file:
                            </ul>
                            """,
                            width=370
                        ),
                        pn.Column(
                            "",
                            group_name_widgets,
                            pn.Row(
                                file_name_input_widgets,
                                download_widget
                            )
                        )
                    ),
                    pn.Spacer(
                        width=950,
                        height=1,
                        background="lightgrey"
                    ),
                    similar_events_tabs,
                    width=1000
                )
            )
        )

        # Build a dictionary to cache the loaded CSV data of similar events
        similar_events_dict = dict()

        # Build the dynamic event page for the selected similar events
        # Dynamic in the sense that it depends on
        # the selected similar events by the user on the Tabulator object
        # and the clicking of the "display_similar_events_button"
        @pn.depends(display_similar_events_button)
        def build_similar_events(_):
            """
            This method returns the dynamic event page(s) for the selected similar
            event(s).
            It depends on the similar events selected by the user on the 
            Tabulator object, and the clicking of the display button.

            :return: the dynamic event page(s) that show(s) the event data, and
                the event waveforms of the selected similar event(s)
            """
            similar_events_selection = similar_events_tabs.selection

            if bool(similar_events_selection):
                similar_events_df_copy = similar_events_df.reset_index()

                similar_events_row = pn.Row()
                similar_events_row.append(pn.Spacer(
                    background="lightgrey",
                    width=1,
                    height=1260
                    )
                )
                similar_events_row.append(build_event_page(
                    selection, 
                    selected_df, 
                    x_axis, 
                    y_axis, 
                    clusters
                    )
                )

                for i in range(len(similar_events_selection)):
                    target_index = similar_events_selection[i]
                    target_similar_event_df = similar_events_df_copy.iloc[target_index]
                    target_similar_event_id = similar_events_df_copy.iloc[
                        target_index,
                        similar_events_df_copy.columns.get_loc("event_id")
                    ]

                    # Event page element - header
                    target_similar_event_header = "#### Similar event number " + str(i + 1) + ":"
                    # Read the event from the cache dictionary if it is there
                    # Otherwise load its CSV file
                    if target_similar_event_id in similar_events_dict:
                        similar_events_row.append(pn.Spacer(
                            background="lightgrey",
                            width=1,
                            height=1260
                            )
                        )
                        similar_events_row.append(pn.Column(
                            target_similar_event_header,
                            similar_events_dict.get(target_similar_event_id)[0],
                            similar_events_dict.get(target_similar_event_id)[1]
                            )
                        )
                    else:
                        # Read the target event CSV file
                        target_similar_event_csv_filename = \
                            os.getcwd() + os.sep + "event_data" + os.sep + \
                            target_similar_event_df["input_event_csv_filename"] + ".csv"
                        target_similar_event_details_df = \
                            pd.read_csv(target_similar_event_csv_filename, header=0)

                        # Extract the event data
                        target_similar_event_start_time = \
                            target_similar_event_details_df["start_time"].values[0]
                        target_similar_event_asset_name = \
                            target_similar_event_details_df["asset_name"].values[0]
                        target_similar_event_x_value = target_similar_event_df[x_axis]
                        target_similar_event_y_value = target_similar_event_df[y_axis]
                        target_similar_event_predictions_dict = \
                            identify_top_predictions(
                                pd.concat([metadata_df, df], axis=1), 
                                target_similar_event_id
                            )

                        # Build the event data as a Row
                        target_similar_event_data = pn.Row(
                            pn.Column(
                                f"""
                                ** _Event metadata:_ **
                                <ul>
                                <li> event_id: {str(target_similar_event_id)}
                                <li> start_time: {str(target_similar_event_start_time)}
                                <li> asset_name: {str(target_similar_event_asset_name)}
                                """,
                                width=300
                            ),
                            pn.Spacer(
                                background="lightgrey",
                                width=1,
                                height=110
                            ),
                            pn.Column(
                                f"""
                                ** _Axis values:_ **
                                <ul>
                                <li> {str(x_axis)}: {"%.5f" % target_similar_event_x_value}
                                <li> {str(y_axis)}: {"%.5f" % target_similar_event_y_value}
                                """,
                                width=300
                            )
                        )
                        if clusters != None:
                            target_similar_event_data.append(
                                pn.Spacer(
                                    background="lightgrey",
                                    width=1,
                                    height=110
                                )
                            )
                            target_similar_event_data.append(
                                pn.Column(
                                    f"""
                                    ** _Cluster value:_ **
                                    <ul>
                                    <li> cluster: {str(target_similar_event_df["cluster"])}
                                    """,
                                )
                            )
                        target_similar_event_waveform_predictons = pn.Row(
                            pn.Column(
                                f"""
                                ** _Top PQD predictions for event's voltages:_ **
                                <ul>
                                <li> Vab: {target_similar_event_predictions_dict["Vab"][0]} 
                                    ({"%.5f" % target_similar_event_predictions_dict["Vab"][1]})
                                <li> Vbc: {target_similar_event_predictions_dict["Vbc"][0]} 
                                    ({"%.5f" % target_similar_event_predictions_dict["Vbc"][1]})
                                <li> Vca: {target_similar_event_predictions_dict["Vca"][0]} 
                                    ({"%.5f" % target_similar_event_predictions_dict["Vca"][1]})
                                </ul>
                                """,
                                width=300
                            ),
                            pn.Spacer(
                                background="lightgrey",
                                width=1,
                                height=110
                            ),
                            pn.Column(
                                f"""
                                ** _Top PQD predictions for event's currents:_ **
                                <ul>
                                <li> Ia: {target_similar_event_predictions_dict["Ia"][0]} 
                                    ({"%.5f" % target_similar_event_predictions_dict["Ia"][1]})
                                <li> Ib: {target_similar_event_predictions_dict["Ib"][0]} 
                                    ({"%.5f" % target_similar_event_predictions_dict["Ib"][1]})
                                <li> Ic: {target_similar_event_predictions_dict["Ic"][0]} 
                                    ({"%.5f" % target_similar_event_predictions_dict["Ic"][1]})
                                </ul>
                                """,
                                width=300
                            )
                        )

                        # Plot the target event waveforms
                        voltages_waveforms = target_similar_event_details_df.hvplot.line(
                            x='timestamps (in ms)',
                            y=['Vab', 'Vbc', 'Vca'],
                            legend=True,
                            responsive=False,
                            height=waveform_height,
                            width=waveform_width,
                            title="Voltages"
                        ).opts(
                            fontscale=waveform_fontsize
                        )
                        currents_waveforms = target_similar_event_details_df.hvplot.line(
                            x='timestamps (in ms)',
                            y=['Ia', 'Ib', 'Ic'],
                            legend=True,
                            responsive=False,
                            height=waveform_height,
                            width=waveform_width,
                            title="Currents"
                        ).opts(
                            fontscale=waveform_fontsize
                        )
                        # Add the target event elements
                        # Event page element - data
                        target_similar_event_data_box = pn.WidgetBox(
                            "#### Event data:",
                            target_similar_event_data,
                            pn.Spacer(
                                background="lightgrey",
                                width=975,
                                height=1
                            ),
                            target_similar_event_waveform_predictons,
                            width=1000
                        )
                        # Event page element - waveforms
                        target_similar_event_waveforms_box = pn.WidgetBox(
                            "#### Event waveforms:",
                            voltages_waveforms,
                            currents_waveforms,
                            width=1000
                        )
                        similar_events_row.append(pn.Spacer(
                            background="lightgrey",
                            width=1,
                            height=1260
                        )
                        )
                        similar_events_row.append(pn.Column(
                            target_similar_event_header,
                            target_similar_event_data_box,
                            target_similar_event_waveforms_box
                        )
                        )
                        # Store the elements in the cache dictionary
                        similar_events_dict[target_similar_event_id] = \
                            [target_similar_event_data_box, target_similar_event_waveforms_box]
                return similar_events_row
        similar_event_page.append(build_similar_events)

        # Return the similar event page
        return similar_event_page

    # Build the dynamic data-exploration pane
    # Essentially an interactive scatter-plot that depends on
    # the parameters selected on the sidebar
    @pn.depends(dr_selection.param.value, clustering_selection.param.value,
                basic_df_x_axis_selection.param.value, basic_df_y_axis_selection.param.value,
                pca_df_x_axis_selection.param.value, pca_df_y_axis_selection.param.value,
                pca_whiten.param.value, pca_svd_solver.param.value,
                umap_df_x_axis_selection.param.value, umap_df_y_axis_selection.param.value,
                tsne_df_x_axis_selection.param.value, tsne_df_y_axis_selection.param.value,
                umap_n_neighbors.param.value, umap_min_dist.param.value,
                tsne_perplexity.param.value, tsne_early_exaggeration.param.value, tsne_learning_rate.param.value,
                k_means_n_clusters_selection.param.value,
                dbscan_max_distance_selection.param.value, dbscan_n_samples_selection.param.value,
                agg_clustering_n_clusters_selection.param.value, agg_clustering_linkage_selection.param.value,
                basic_similar_events_x_value_selection.param.value,
                basic_similar_events_y_value_selection.param.value,
                dr_similar_events_x_value_selection.param.value,
                dr_similar_events_y_value_selection.param.value)
    def data_exploration(dr_value, clustering_value,
                         basic_x_value, basic_y_value,
                         pca_x_value, pca_y_value,
                         pca_whiten_value, pca_svd_solver_value,
                         umap_x_value, umap_y_value,
                         tsne_x_value, tsne_y_value,
                         umap_n_neighbors_value, umap_min_dist_value,
                         tsne_perplexity_value, tsne_early_exaggeration_value, tsne_learning_rate_value,
                         k_means_n_clusters,
                         dbscan_max_distance_value, dbscan_n_samples_value,
                         agg_clustering_n_clusters, agg_clustering_linkage_value,
                         basic_similar_events_x_value, basic_similar_events_y_value,
                         dr_similar_events_x_value, dr_similar_events_y_value):
        """
        This method builds the dynamic data-exploration pane.
        It depends on the parameters selected on the sidebar of the dashboard.

        :return: the dynamic data-exploration pane that renders the
            scatter-plot of the loaded events
        """

        # Option A: No dimensionality reduction nor clustering algos
        if dr_value == "Nil" and clustering_value == "Nil":
            if basic_x_value == basic_y_value:
                selected_df = basic_df[[basic_x_value]]
            else:
                selected_df = basic_df[[basic_x_value, basic_y_value]]
            selected_df = pd.concat([metadata_df, selected_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=alt.X(basic_x_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                y=alt.Y(basic_y_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", basic_x_value, basic_y_value]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            # Build the interactive event tab
            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        None
                    )
            # Build the interactive similar events tab
            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        None,
                        basic_similar_events_x_value,
                        basic_similar_events_y_value
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option B: PCA and no clustering algo
        elif dr_value == "PCA" and clustering_value == "Nil":
            pca = decomposition.PCA(
                n_components=2,
                whiten=pca_whiten_value,
                svd_solver=pca_svd_solver_value
            )
            pca.fit(df)
            pca_data = pca.transform(df)
            pca_df = pd.DataFrame(pca_data, columns=pca_labels)

            selected_df = pd.concat([metadata_df, pca_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=pca_x_value,
                y=pca_y_value,
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", pca_x_value, pca_y_value]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        None
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        None,
                        dr_similar_events_x_value,
                        dr_similar_events_y_value
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option C: UMAP and no clustering algo
        elif dr_value == "UMAP" and clustering_value == "Nil":
            umap_reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors_value,
                min_dist=umap_min_dist_value,
                random_state=0
            )
            umap_reducer.fit(df)
            umap_data = umap_reducer.transform(df)
            umap_df = pd.DataFrame(umap_data, columns=umap_labels)

            selected_df = pd.concat([metadata_df, umap_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=umap_x_value,
                y=umap_y_value,
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", umap_x_value, umap_y_value]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        None
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        None,
                        dr_similar_events_x_value,
                        dr_similar_events_y_value
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Opton D: No dimensionality reduction and K-Means clustering algo
        elif dr_value == "Nil" and clustering_value == "K-Means":
            basic_kmeans = KMeans(n_clusters=k_means_n_clusters)
            if basic_x_value == basic_y_value:
                selected_df = basic_df[[basic_x_value]]
            else:
                selected_df = basic_df[[basic_x_value, basic_y_value]]
            y_pred = basic_kmeans.fit_predict(selected_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, selected_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=alt.X(basic_x_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                y=alt.Y(basic_y_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", basic_x_value, basic_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option E: PCA and K-Means
        elif dr_value == "PCA" and clustering_value == "K-Means":
            pca = decomposition.PCA(
                n_components=2,
                whiten=pca_whiten_value,
                svd_solver=pca_svd_solver_value
            )
            pca.fit(df)
            pca_data = pca.transform(df)
            pca_df = pd.DataFrame(pca_data, columns=pca_labels)

            pca_kmeans = KMeans(n_clusters=k_means_n_clusters)
            y_pred = pca_kmeans.fit_predict(pca_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, pca_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=pca_x_value,
                y=pca_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", pca_x_value, pca_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option F: UMAP and K-Means
        elif dr_value == "UMAP" and clustering_value == "K-Means":
            umap_reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors_value,
                min_dist=umap_min_dist_value,
                random_state=0
            )
            umap_reducer.fit(df)
            umap_data = umap_reducer.transform(df)
            umap_df = pd.DataFrame(umap_data, columns=umap_labels)

            umap_kmeans = KMeans(n_clusters=k_means_n_clusters)
            y_pred = umap_kmeans.fit_predict(umap_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, umap_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=umap_x_value,
                y=umap_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", umap_x_value, umap_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option G: No dimensionality reduction algo and DBSCAN clustering algo
        elif dr_value == "Nil" and clustering_value == "DBSCAN":
            basic_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
            if basic_x_value == basic_y_value:
                selected_df = basic_df[[basic_x_value]]
            else:
                selected_df = basic_df[[basic_x_value, basic_y_value]]
            y_pred = basic_dbscan.fit_predict(selected_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, selected_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=alt.X(basic_x_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                y=alt.Y(basic_y_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", basic_x_value, basic_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option H: PCA and DBSCAN
        elif dr_value == "PCA" and clustering_value == "DBSCAN":
            pca = decomposition.PCA(
                n_components=2,
                whiten=pca_whiten_value,
                svd_solver=pca_svd_solver_value
            )
            pca.fit(df)
            pca_data = pca.transform(df)
            pca_df = pd.DataFrame(pca_data, columns=pca_labels)

            pca_dbscan = DBSCAN(
                eps=dbscan_max_distance_value, 
                min_samples=dbscan_n_samples_value
            )
            y_pred = pca_dbscan.fit_predict(pca_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, pca_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=pca_x_value,
                y=pca_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", pca_x_value, pca_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option I: UMAP and DBSCAN
        elif dr_value == "UMAP" and clustering_value == "DBSCAN":
            umap_reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors_value,
                min_dist=umap_min_dist_value,
                random_state=0
            )
            umap_reducer.fit(df)
            umap_data = umap_reducer.transform(df)
            umap_df = pd.DataFrame(umap_data, columns=umap_labels)

            umap_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
            y_pred = umap_dbscan.fit_predict(umap_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, umap_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=umap_x_value,
                y=umap_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", umap_x_value, umap_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option J: TSNE and no clustering algo
        elif dr_value == "t-SNE" and clustering_value == "Nil":
            tsne = TSNE(
                perplexity=tsne_perplexity_value,
                early_exaggeration=tsne_early_exaggeration_value,
                learning_rate=tsne_learning_rate_value,
                random_state=0
            )
            tsne_data = tsne.fit_transform(df)
            tsne_df = pd.DataFrame(tsne_data, columns=tsne_labels)

            selected_df = pd.concat([metadata_df, tsne_df], axis=1)
            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=tsne_x_value,
                y=tsne_y_value,
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", tsne_x_value, tsne_y_value]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        None
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        None,
                        basic_similar_events_x_value,
                        basic_similar_events_y_value
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option K: t-SNE and K-Means
        elif dr_value == "t-SNE" and clustering_value == "K-Means":
            tsne = TSNE(
                perplexity=tsne_perplexity_value,
                early_exaggeration=tsne_early_exaggeration_value,
                learning_rate=tsne_learning_rate_value,
                random_state=0
            )
            tsne_data = tsne.fit_transform(df)
            tsne_df = pd.DataFrame(tsne_data, columns=tsne_labels)

            tsne_kmeans = KMeans(n_clusters=k_means_n_clusters)
            y_pred = tsne_kmeans.fit_predict(tsne_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, tsne_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=tsne_x_value,
                y=tsne_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", tsne_x_value, tsne_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option L: t-SNE and DBSCAN
        elif dr_value == "t-SNE" and clustering_value == "DBSCAN":
            tsne = TSNE(
                perplexity=tsne_perplexity_value,
                early_exaggeration=tsne_early_exaggeration_value,
                learning_rate=tsne_learning_rate_value,
                random_state=0
            )
            tsne_data = tsne.fit_transform(df)
            tsne_df = pd.DataFrame(tsne_data, columns=tsne_labels)

            tsne_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
            y_pred = tsne_dbscan.fit_predict(tsne_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, tsne_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=tsne_x_value,
                y=tsne_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", tsne_x_value, tsne_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option M: No dimensionality reduction algo and Agglomerative Clustering
        elif dr_value == "Nil" and clustering_value == "Agglomerative Clustering":
            basic_agg_clustering = AgglomerativeClustering(
                n_clusters=agg_clustering_n_clusters,
                linkage=agg_clustering_linkage_value
            )
            if basic_x_value == basic_y_value:
                selected_df = basic_df[[basic_x_value]]
            else:
                selected_df = basic_df[[basic_x_value, basic_y_value]]
            y_pred = basic_agg_clustering.fit_predict(selected_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, selected_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=alt.X(basic_x_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                y=alt.Y(basic_y_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", basic_x_value, basic_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        basic_x_value,
                        basic_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option N: t-SNE and Agglomerative Clustering
        elif dr_value == "t-SNE" and clustering_value == "Agglomerative Clustering":
            tsne = TSNE(
                perplexity=tsne_perplexity_value,
                early_exaggeration=tsne_early_exaggeration_value,
                learning_rate=tsne_learning_rate_value,
                random_state=0
            )
            tsne_data = tsne.fit_transform(df)
            tsne_df = pd.DataFrame(tsne_data, columns=tsne_labels)

            tsne_agg_clustering = AgglomerativeClustering(
                n_clusters=agg_clustering_n_clusters,
                linkage=agg_clustering_linkage_value
            )
            y_pred = tsne_agg_clustering.fit_predict(tsne_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, tsne_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=tsne_x_value,
                y=tsne_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", tsne_x_value, tsne_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        tsne_x_value,
                        tsne_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option O: PCA and Agglomerative Clustering
        elif dr_value == "PCA" and clustering_value == "Agglomerative Clustering":
            pca = decomposition.PCA(
                n_components=2,
                whiten=pca_whiten_value,
                svd_solver=pca_svd_solver_value
            )
            pca.fit(df)
            pca_data = pca.transform(df)
            pca_df = pd.DataFrame(pca_data, columns=pca_labels)

            pca_agg_clustering = AgglomerativeClustering(
                n_clusters=agg_clustering_n_clusters,
                linkage=agg_clustering_linkage_value
            )
            y_pred = pca_agg_clustering.fit_predict(pca_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, pca_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=pca_x_value,
                y=pca_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", pca_x_value, pca_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        pca_x_value,
                        pca_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )

        # Option P: UMAP and Agglomerative Clustering
        elif dr_value == "UMAP" and clustering_value == "Agglomerative Clustering":
            # UMAP
            umap_reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors_value,
                min_dist=umap_min_dist_value,
                random_state=0
            )
            umap_reducer.fit(df)
            umap_data = umap_reducer.transform(df)
            umap_df = pd.DataFrame(umap_data, columns=umap_labels)

            umap_agg_clustering = AgglomerativeClustering(
                n_clusters=agg_clustering_n_clusters,
                linkage=agg_clustering_linkage_value
            )
            y_pred = umap_agg_clustering.fit_predict(umap_df)

            y_pred_df = pd.DataFrame(data={"cluster": y_pred})
            selected_df = pd.concat([metadata_df, umap_df, y_pred_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=umap_x_value,
                y=umap_y_value,
                color=alt.condition(
                    selector,
                    alt.Color('cluster:N', scale=alt.Scale(scheme='set1'), legend=None),
                    alt.value('lightgray')),
                tooltip=["event_id", umap_x_value, umap_y_value, "cluster"]
            ).properties(
                height=data_exploration_pane_height,
                width=data_exploration_pane_width
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        clustering_value
                    )

            def get_similar_events(selection):
                if not selection:
                    return '### No event selection'
                else:
                    return build_similar_event_page(
                        selection,
                        selected_df,
                        umap_x_value,
                        umap_y_value,
                        clustering_value,
                        None,
                        None
                    )

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Selected event", pn.bind(get_event, vega_pane.selection.param.event_id)),
                ("Similar events", pn.bind(get_similar_events, vega_pane.selection.param.event_id))
            )


    ## ========================================================
    ## Return the dashboard's dynamic panes
    return pn.Row(
        pn.Column(
            algo_column,
            plot_configuration,
            similar_events_configuration
        ),
        pn.Column(
            data_exploration
        )
    )


## ========================================================
## Add the dynamic panes to the dashboard
app.main.append(dynamic_env)
app.servable()
