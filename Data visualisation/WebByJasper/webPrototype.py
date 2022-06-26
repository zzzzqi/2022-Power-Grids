import io
import random
import pandas
import panel as pn
import numpy as np
import pandas as pd
import altair as alt
import holoviews as hv
import hvplot.pandas  # noqa
from hvplot import hvPlot
from bokeh.resources import INLINE
from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

from sklearn import decomposition
from sklearn.cluster import KMeans, DBSCAN, k_means
import umap.umap_ as umap

# Enable Bokeh and Panel
hv.extension('bokeh')
pn.extension('vega')

## ========================================================
## Build the layout of the dashboard
# Set the template used in the dashboard
app = pn.template.MaterialTemplate(title='Smart Power Grids Dashboard')
# Set the dashboard's instructions
instructions = """
    #### Instructions: <br>
    <ol>
    <li> Upload the CNN output to the dashboard <br>
    <li> Choose the Dimensionality Reduction algo and the corresponding variables to plot the data-exploration pane <br>
    <li> Choose the Clustering algo to perform unspervised grouping <br>
    <li> Select a dot from the plot. Open a window with the followings: <br>
    <ul>
    <li> This event's 6 waveforms (3 voltages and 3 currents)
    <li> Similar events to this
    <li> The 6 waveforms of one of the similar events
    </ul>
    <li> Perform grouping and labelling on the plot <br>
    <li> Export the data of the grouped events in CSV <br>
    </ol>
    """
file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
file_input_message = "#### Upload the CNN output: <br>"
# Set the dashboard's sidebar
app.sidebar.append(instructions)
app.sidebar.append(file_input_message)
app.sidebar.append(file_input)


## ========================================================
## Read the input CSV file and set it for the dynamic environment
@pn.depends(file_input)
def dynamic_env(df):
    if df is None:
        metadata_columns = ["metadata_dummy_axis" + str(i) for i in range(0, 5)]
        metadata_columns[0] = "event_id"
        metadata_df = pd.DataFrame(np.random.uniform(0, 1, size=(100, 5)), columns=metadata_columns)
        df = pd.DataFrame(np.random.uniform(0, 1, size=(100, 60)),
                          columns=["dummy_axis" + str(i) for i in range(0, 60)])
    else:
        metadata_df = pd.read_csv(io.BytesIO(file_input.value), header=0, usecols=range(1, 5))
        df = pd.read_csv(io.BytesIO(file_input.value), header=0, usecols=range(6, 66))
    metadata_df = metadata_df.dropna()
    df = df.dropna()

    ## ========================================================
    ## Basic dataframe
    basic_df = pd.DataFrame(df)
    basic_df_headers = list(basic_df.columns)

    ## ========================================================
    ## Dimensionality Reduction algos
    # PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(df)
    pca_data = pca.transform(df)
    pca_labels = ["PC 1", "PC 2"]
    pca_df = pd.DataFrame(pca_data, columns=pca_labels)
    pca_df_headers = list(pca_df.columns)
    # UMAP
    umap_reducer = umap.UMAP()
    umap_reducer.fit(df)
    umap_data = umap_reducer.transform(df)
    umap_labels = ["UMAP 1", "UMAP 2"]
    umap_df = pd.DataFrame(umap_data, columns=umap_labels)

    ## ========================================================
    ## Column 1a: The algo-selection column
    # Build the selection dropdown for Dimensionaity Reduction algos
    dr_selection = pn.widgets.Select(
        options=["Nil", "PCA", "UMAP"]
    )
    # Build the selection dropdown for Clustering algos
    clustering_selection = pn.widgets.Select(
        options=["Nil", "K-Means", "DBSCAN"]
    )
    # Build the WidgetBox for algos selection
    algo_column = pn.WidgetBox(
        pn.pane.Markdown("#### Dimensionality reduction algos: "),
        dr_selection,
        pn.pane.Markdown("#### Clustering algos: "),
        clustering_selection,
        pn.pane.Markdown("")
    )

    ## ========================================================
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
        value=10,
        start=5,
        end=50,
        step=5,
        name="Number of samples in a neighbourhood"
    )

    # Build the WidgetBox for plot configuration
    @pn.depends(dr_selection.param.value, clustering_selection.param.value)
    def plot_configuration(dr_value, clustering_value):
        if dr_value == "Nil" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "PCA" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "UMAP" and clustering_value == "Nil":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "Nil" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                k_means_n_clusters_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "PCA" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                k_means_n_clusters_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "UMAP" and clustering_value == "K-Means":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                k_means_n_clusters_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "Nil" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                basic_df_x_axis_selection,
                basic_df_y_axis_selection,
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "PCA" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                pca_df_x_axis_selection,
                pca_df_y_axis_selection,
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown("")
            )
        elif dr_value == "UMAP" and clustering_value == "DBSCAN":
            return pn.WidgetBox(
                pn.pane.Markdown("#### Data-exploration pane options: "),
                umap_df_x_axis_selection,
                umap_df_y_axis_selection,
                dbscan_max_distance_selection,
                dbscan_n_samples_selection,
                pn.pane.Markdown("")
            )

    ## ========================================================
    ## Column 2: The data-exploration pane

    # Build the event_page

    Head_Selected_Waveform = """
    ###    Six Waveforms of Selected Event :
    """

    Head_Selected_Details = """
    ###    Details of Selected Event :
    """
    Head_Similar_Events = """
        ###    Similar events :
        """

    df_event_page = pd.read_csv("flickers_sample00.csv", header=0, engine='python')

    event_id_1 = pn.widgets.StaticText(value='event_id:', width=50, align='center')
    event_id_2 = pn.widgets.StaticText(value='001', width=10, align='center')
    start_time_1 = pn.widgets.StaticText(value='start_time:', width=60, align='center')
    start_time_2 = pn.widgets.StaticText(value='2022_06_01', width=60, align='center')
    asset_name_1 = pn.widgets.StaticText(value='asset_name:', width=70, align='center')
    asset_name_2 = pn.widgets.StaticText(value='test_000', width=70, align='center')

    Layout_details = pn.Column(Head_Selected_Details,
                               pn.Row(event_id_1, event_id_2, start_time_1, start_time_2, asset_name_1, asset_name_2))

    event_page_event_detail = pn.WidgetBox(Layout_details, width=860)

    Vab_waveform = df_event_page.hvplot.line(
        x='timestamps (in ms)',
        y='Vab',
        legend=False,
        responsive=False,
        height=80,
        width=850,
        title="Vab_waveform of Selected event"
    )

    Vbc_waveform = df_event_page.hvplot.line(
        x='timestamps (in ms)',
        y='Vbc',
        legend=True,
        responsive=False,
        height=80,
        width=850,
        title="Vbc_waveform of Selected event"
    )

    Vca_waveform = df_event_page.hvplot.line(
        x='timestamps (in ms)',
        y='Vca',
        legend=True,
        responsive=False,
        height=80,
        width=850,
        title="Vca_waveform of Selected event"
    )

    Ia_waveform = df_event_page.hvplot.line(
        x='timestamps (in ms)',
        y='Ia',
        legend=True,
        responsive=False,
        height=80,
        width=850,
        title="Ia_waveform of Selected event"
    )

    Ib_waveform = df_event_page.hvplot.line(
        x='timestamps (in ms)',
        y='Ib',
        legend=True,
        responsive=False,
        height=80,
        width=850,
        title="Ib_waveform of Selected event"
    )

    Ic_waveform = df_event_page.hvplot.line(
        x='timestamps (in ms)',
        y='Ic',
        legend=True,
        responsive=False,
        height=80,
        width=850,
        title="Ic_waveform of Selected event"
    )

    fontsize_value = '90%'
    Vab_waveform.opts(
        xaxis=None,
        fontsize={
            'title': fontsize_value,
            'labels': fontsize_value,
            'ticks': fontsize_value,
        })
    Vbc_waveform.opts(
        xaxis=None,
        fontsize={
            'title': fontsize_value,
            'labels': fontsize_value,
            'ticks': fontsize_value,
        })
    Vca_waveform.opts(xaxis=None,
                      fontsize={
                          'title': fontsize_value,
                          'labels': fontsize_value,
                          'ticks': fontsize_value,
                      })
    Ia_waveform.opts(xaxis=None,
                     fontsize={
                         'title': fontsize_value,
                         'labels': fontsize_value,
                         'ticks': fontsize_value,
                     })
    Ib_waveform.opts(xaxis=None,
                     fontsize={
                         'title': fontsize_value,
                         'labels': fontsize_value,
                         'ticks': fontsize_value,
                     })
    Ic_waveform.opts(fontsize={
        'title': fontsize_value,
        'labels': fontsize_value,
        'ticks': fontsize_value,
    })

    df_similar_event = pd.DataFrame({
        'Event_Id': ['Event ' + str(x) for x in range(0, 10)],
        'Sag': [bool(random.randint(0, 1)) for _ in range(10)],
        'Swell': [bool(random.randint(0, 1)) for _ in range(10)],
        'Interruption': [bool(random.randint(0, 1)) for _ in range(10)],
        'Flicker': [bool(random.randint(0, 1)) for _ in range(10)],
        'Harmonic': [bool(random.randint(0, 1)) for _ in range(10)],
        'Oscillatory Transient': [bool(random.randint(0, 1)) for _ in range(10)],
        'Spike': [bool(random.randint(0, 1)) for _ in range(10)],
        'Sag & Harmonic': [bool(random.randint(0, 1)) for _ in range(10)],
        'Interruption & Harmonic': [bool(random.randint(0, 1)) for _ in range(10)],
        'Swell & Harmonic': [bool(random.randint(0, 1)) for _ in range(10)],
    })

    bokeh_formatters = {
        'Sag': BooleanFormatter(),
        'Swell': BooleanFormatter(),
        'Interruption': BooleanFormatter(),
        'Flicker': BooleanFormatter(),
        'Harmonic': BooleanFormatter(),
        'Oscillatory Transient': BooleanFormatter(),
        'Spike': BooleanFormatter(),
        'Sag & Harmonic': BooleanFormatter(),
        'Interruption & Harmonic': BooleanFormatter(),
        'Swell & Harmonic': BooleanFormatter(),
    }

    similar_event_widget = pn.widgets.Tabulator(df_similar_event, formatters=bokeh_formatters,
                                                show_index=False,
                                                widths={'Event_Id': '12%', 'Sag': '8%', 'Swell': '8%',
                                                        'Interruption': '9%', 'Flicker': '9%',
                                                        'Harmonic': '9%', 'Oscillatory Transient': '9%', 'Spike': '9%',
                                                        'Sag & Harmonic': '9%', 'Interruption & Harmonic': '9%',
                                                        'Swell & Harmonic': '9%', },
                                                sizing_mode='stretch_width')

    event_page_layout = pn.Column(pn.Column(event_page_event_detail,
                                            pn.WidgetBox(pn.Column(Head_Selected_Waveform,
                                                                   Vab_waveform, Vbc_waveform, Vca_waveform,
                                                                   Ia_waveform, Ib_waveform, Ic_waveform))),
                                  pn.WidgetBox(Head_Similar_Events, similar_event_widget, width=860))

    # ///////////////////////////////////////////
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
        # Option A: No dimensionality reduction nor clustering algos
        if dr_value == "Nil" and clustering_value == "Nil":
            selected_df = basic_df[[basic_x_value, basic_y_value]]
            selected_df = pd.concat([metadata_df, selected_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=alt.X(basic_x_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                y=alt.Y(basic_y_value, scale=alt.Scale(domain=[-0.1, 1.1])),
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", basic_x_value, basic_y_value]
            ).properties(
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option B: PCA and no clustering algo
        elif dr_value == "PCA" and clustering_value == "Nil":
            selected_df = pd.concat([metadata_df, pca_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=pca_x_value,
                y=pca_y_value,
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", pca_x_value, pca_y_value]
            ).properties(
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option C: UMAP and no clustering algo
        elif dr_value == "UMAP" and clustering_value == "Nil":
            selected_df = pd.concat([metadata_df, umap_df], axis=1)

            selector = alt.selection_single(name='event_id')
            plot = alt.Chart(selected_df).mark_circle(size=80).encode(
                x=umap_x_value,
                y=umap_y_value,
                color=alt.condition(selector, alt.value("navy"), alt.value('lightgray')),
                tooltip=["event_id", umap_x_value, umap_y_value]
            ).properties(
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Opton D: No dimensionality reduction and K-Means clustering algo
        elif dr_value == "Nil" and clustering_value == "K-Means":
            basic_kmeans = KMeans(n_clusters=k_means_n_clusters)
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
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            # Add overlay for centroids
            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option E: PCA and K-Means
        elif dr_value == "PCA" and clustering_value == "K-Means":
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
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            # Add overlay for centroids
            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option F: UMAP and K-Means
        elif dr_value == "UMAP" and clustering_value == "K-Means":
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
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            # Add overlay for centroids
            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option G: No dimensionality reduction algo and DBSCAN clustering algo
        elif dr_value == "Nil" and clustering_value == "DBSCAN":
            basic_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
            selected_df = basic_df[[basic_x_value, basic_y_value]]
            y_pred = basic_dbscan.fit_predict(selected_df)
            labels = basic_dbscan.labels_

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
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            # Add overlay for centroids
            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option H: PCA and DBSCAN
        elif dr_value == "PCA" and clustering_value == "DBSCAN":
            pca_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
            y_pred = pca_dbscan.fit_predict(pca_df)
            labels = pca_dbscan.labels_

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
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            # Add overlay for centroids
            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )
        # Option I: UMAP and DBSCAN
        elif dr_value == "UMAP" and clustering_value == "DBSCAN":
            umap_dbscan = DBSCAN(eps=dbscan_max_distance_value, min_samples=dbscan_n_samples_value)
            y_pred = umap_dbscan.fit_predict(umap_df)
            labels = umap_dbscan.labels_

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
                height=630,
                width=700,
                title="Data-exploraton pane"
            ).interactive().add_selection(selector)
            vega_pane = pn.pane.Vega(plot, debounce=10)

            def get_event(selection):
                if not selection:
                    return '## No selection'
                else:
                    return selected_df.iloc[selection[0] - 1]

            # Add overlay for centroids
            return pn.Tabs(
                ("Data-exploration pane", vega_pane),
                ("Power signal event", pn.Column(
                    event_page_layout
                )
                 )
            )

    ## ========================================================
    ## Return the dashboard's dynamic panes
    return pn.Row(
        pn.Column(
            algo_column,
            plot_configuration
        ),
        pn.Column(
            data_exploration
        )
    )


## ========================================================
## Add the dynamic panes to the dashboard
app.main.append(dynamic_env)
app.servable()
# app.save('test_interactive_script.html', resources=INLINE)
