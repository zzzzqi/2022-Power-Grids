.. Smart Power Grid documentation master file, created by
   sphinx-quickstart on 13 Aug 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Project documentation
=====================

Installing dependencies for this project:
-----------------------------------------
This is how to install all the required dependencies for this project.
::
   pip install -r requirements.txt

Part One: Input Handling Tool
----------------------------------
This is a tool to read power events, analyse their signal waveforms, 
and classify them according to IEEE standards, enabled by the use
of a Convolutional Neural Network (CNN).

Link to the `Part One script`_.

.. _Part One script: https://github.com/zzzzqi/2022-Power-Grids/blob/main/Production/final/input_handling_tool.py/

How to use the tool:
^^^^^^^^^^^^^^^^^^^^
A. Operating logic:
"""""""""""""""""""
The tool would read input events stored in the sub-directory "event_data".
It would use the trained CNN model stored in the sub-directory 
"trained_cnn_models". By default, the model used is "basic_pqd_cnn.h5".
After running the tool, it would create a sub-directory 
"event_waveform_images" to store the generated 2D PSR images 
(if the no images option is not chosen), and generate an output CSV file 
"cnn_output.csv".

B. Preparation:
"""""""""""""""
After installing the "click" library, run the following command 
in the folder where the tool is located. 
::
   pip install --editable .

C. Usage:
"""""""""
These are the instructions on how to use the the tool. 
::
   Use: 
      handleinput [OPTIONS] FILEPATH 
   Help: 
      handleinput --help

D. Examples:
""""""""""""
Examples of using the tool are as follows.

*Converting input events into 2D PSR images:*
::
   handleinput --convert TRUE FILEPATH

*Making predictions on the generated 2D PSR images:*
::
   handleinput --predict TRUE FILEPATH

*Making predictions on the input events with NO images saved (If this option is enabled, 
then the two options above will be disabled):*
::
   handleinput --no_images TRUE FILEPATH
   which equals to:
   handleinput --no_images TRUE --convert TRUE FILEPATH
   handleinput --no_images TRUE --predict TRUE FILEPATH

*Possible to simply use "." to use the local directory as the FILEPATH:*
::
   handleinput --convert TRUE .
   handleinput --predict TRUE .
   handleinput --no_images TRUE .

Methods:
^^^^^^^^
The details of all the methods used in the Part One script are as follows: 

.. py:function:: input_handling_tool.main(filepath, convert, predict, output_name, noimages):

   This method executes the script according to the selected parameters.

   :param filepath: the filepath of the script
   :param convert: the command line option to convert the signals 
      to 2D PSR images
   :param predict: the command line option to make predictions on the 
      PQD types of each signal
   :param output_name: the name of the output CSV file
   :param noimages: the command line option to make predictions on input 
      events with NO images saved

|

.. py:function:: input_handling_tool.convert_signals(
      input_event_dir, psr_dir, output_csv_filepath):

   This method converts input signals into the 2D PSR images.
   This is one of the command line tool options.

   :param input_event_dir: the directory of the input event CSV files
   :param psr_dir: the directory of the 2D PSR images
   :param output_csv_filepath: the directory of the output CSV file to 
      be stored

|

.. py:function:: input_handling_tool.make_predictions(
      cnn_model_path, psr_dir, output_csv_filepath):

   This method uses the CNN model to make predictions on the 2D PSR images, 
   located in the "psr_dir" parameter.
   This is one of the command line tool options.

   :param cnn_model_path: the directory of the CNN model to be used
   :param psr_dir: the directory of the 2D PSR images
   :param output_csv_filepath: the directory of the output CSV file to 
      be stored

|

.. py:function:: input_handling_tool.predict_from_events(
      cnn_model_path, input_event_dir, output_csv_filepath):

   This method uses the CNN model to make predictions on the input events, 
   with NO images saved locally.
   This is one of the command line tool options.

   :param cnn_model_path: the directory of the CNN model to be used
   :param input_event_dir: the directory of the input event CSV files
   :param output_csv_filepath: the directory of the output CSV file to 
      be stored

|

.. py:function:: input_handling_tool.phase_space_graph(
      import_csv, export_path, tau=20):

   This method is the helper function for converting all the six 
   signal waveforms into 2D PSR images.

   :param import_csv: the input event CSV file
   :param export_path: the directory for storing the 2D PSR images
   :param tau: the time lag parameter for PSR, set to be 20 on default

|

.. py:function:: input_handling_tool.identify_max_value(signal):

   This method is the helper function for identifying the normal 
   max value of the signal waveform.
   First it detects the transformations of the waveforms from positive 
   to negative, and vice versa, twice.
   Then it identifies the max value in this specific interval in the 
   waveforms.

   :param signal: the waveform data
   :return: the normal max value of the signal waveform

|

.. py:function:: input_handling_tool.mkdir(path):

   This method is the helper function for creating a specific directory.

   :param path: the specific directory to be created

|

Part Two: Web Dashboard
-----------------------------
This is a web dashboard that reads the output file from the CNN, 
display the events, and employs a selection of Dimensionality Reduction 
and Clustering algorithms to enable interactive data exploration.

Link to the `Part Two script`_.

.. _Part Two script: https://github.com/zzzzqi/2022-Power-Grids/blob/main/Production/final/web_dashboard.py/

How to use the dashboard:
^^^^^^^^^^^^^^^^^^^^^^^^^
A. Operating logic:
""""""""""""""""""""
The web dashboard requires the user to upload the output CSV file generated 
from the Part One input handling tool.
After the dashboard has started running, use the upload button on its 
sidebar to upload the said CSV file to enable data-exploration.

B. Usage:
""""""""""
This is the instruction on how to run the web dashboard.
::
   Use: 
      panel serve web_dashboard.py --show

Methods:
^^^^^^^^
The details of all the methods used in the Part Two script are as follows: 

.. py:function:: web_dashboard.dynamic_env(read_file):

   This method reads the CNN output file as the input 
   and uses it for the dynamic environment of the dashboard.

   :param read_file: the CNN output file
   :return: the dynamic environment of the dashboard

|

.. py:function:: web_dashboard.dynamic_env.data_exploration(
   dr_value, clustering_value,
   basic_x_value, basic_y_value, 
   pca_x_value, pca_y_value, pca_whiten_value, pca_svd_solver_value,
   umap_x_value, umap_y_value, 
   tsne_x_value, tsne_y_value, 
   umap_n_neighbors_value, umap_min_dist_value,
   tsne_perplexity_value, tsne_early_exaggeration_value, 
   tsne_learning_rate_value,
   k_means_n_clusters, 
   dbscan_max_distance_value, dbscan_n_samples_value, 
   agg_clustering_n_clusters, agg_clustering_linkage_value, 
   basic_similar_events_x_value, basic_similar_events_y_value,
   dr_similar_events_x_value, dr_similar_events_y_value):
   
   This method builds the dynamic data-exploration pane.
   It depends on the selected parameters on the sidebar of the dashboard.

   :param dr_value: the selected Dimensionality Reduction algo
   :param clustering_value: the selected Clustering algo
   :param basic_x_value: the selected basic x-axis
   :param basic_y_value: the selected basic x-axis
   :param pca_x_value: the selected x-axis for PCA dataframes
   :param pca_y_value: the selected y-axis for PCA dataframes
   :param pca_whiten_value: the selected whiten value for PCA algo
   :param pca_svd_solver_value: the selected svd solver value 
      for PCA algo
   :param umap_x_value: the selected x-axis for UMAP dataframes
   :param umap_y_value: the selected y-axis for UMAP dataframes
   :param tsne_x_value: the selected x-axis for TSNE dataframes
   :param tsne_y_value: the selected y-axis for TSNE dataframes
   :param umap_n_neighbors_value: the selected n_neighbors value 
      for UMAP algo
   :param umap_min_dist_value: the selected min_dist value 
      for UMAP algo
   :param tsne_perplexity_value: the selected perplexity value 
      for TSNE algo
   :param tsne_early_exaggeration_value: the selected early exaggeration 
      value for TSNE algo
   :param tsne_learning_rate_value: the selected learning rate value 
      for TSNE algo
   :param k_means_n_clusters: the selected number of clusters 
      for K-Means algo
   :param dbscan_max_distance_value: the selected max_distance value 
      for DBSCAN algo
   :param dbscan_n_samples_value: the selected n_samples value 
      for DBSCAN algo
   :param agg_clustering_n_clusters: the selected number of clusters 
      for Agglomerative Clustering
   :param agg_clustering_linkage_value: the selected linakge value 
      for Agglomerative Clustering
   :param basic_similar_events_x_value: the selected interval of x-value 
      for identifying similar events in basic dataframes
   :param basic_similar_events_y_value: the selected interval of y-value 
      for identifying similar events in basic dataframes
   :param dr_similar_events_x_value: the selected interval of x-value 
      for identifying similar events in dataframes 
      with Dimensionality Reduction algos 
   :param dr_similar_events_y_value: the selected interval of y-value 
      for identifying similar events in dataframes 
      with Dimensionality Reduction algos 
   :return: the dynamic data-exploration pane that renders the
      scatter-plot of the loaded events

|

.. py:function:: web_dashboard.dynamic_env.plot_configuration(
   dr_value, clustering_value):
   
   This method builds the dynamic widgetbox for plot configuration.
   It depends on the selected Dimensionality Reduction and Clustering algos.

   :param dr_value: the selected Dimensionality Reduction algo
   :param clustering_value: the selected Clustering algo
   :return: the dynamic widgetbox for plot configuration, listing 
      the axis options for the scatter-plot, and 
      the parameter options for the selected algos

|

.. py:function:: web_dashboard.dynamic_env.similar_events_configuration(
   dr_value, clustering_value):
   
   This method builds the dynamic widgetbox for similar events configuration.
   It depends on the selected Dimensionality Reduction and Clustering algos.

   :param dr_value: the selected Dimensionality Reduction algo
   :param clustering_value: the selected Clustering algo
   :return: the dynamic widgetbox for configuring similar events, listing 
      the options for changing how similar events are identified

|

.. py:function:: web_dashboard.dynamic_env.identify_top_predictions(
   df, event_id):
   
   This method identifies the top PQD predictions of the signal waveforms.

   :param df: the dataframe of the selected event
   :param event_id: the id of the selected event
   :return: the dictionary where waveform names are keys, 
      and the arrays of their top PQD types and prediction scores 
      are the values

|

.. py:function:: web_dashboard.dynamic_env.build_event_page(selection,
   selected_df, x_axis, y_axis, clusters):
   
   This method builds the dynamic event page.
   It depends on the selected event on the data-exploration pane.

   :param selection: the selected event on the data-exploration pane
   :param selected_df: the dataframe of the selected event and 
      its similar events modified by the selected axes and algos 
   :param x_axis: the selected x-axis
   :param y_axis: the selected y-axis
   :param clusters: the selected Clustering algo
   :return: the dynamic event page that shows the event data, and
      the event waveforms of the selected event

|

.. py:function:: web_dashboard.dynamic_env.build_similar_event_page(
   selection, selected_df, x_axis, y_axis, 
   clusters, similar_events_x_parameter, similar_events_y_parameter):
   
   This method builds the dynamic page of similar events.
   It depends on the selected event on the data-exploration pane, and 
   the parameters chosen for configuring the similar event selection.

   :param selection: the selected event on the data-exploration pane
   :param selected_df: the dataframe of the selected event and 
      its similar events modified by the selected axes and algos 
   :param x_axis: the selected x-axis
   :param y_axis: the selected y-axis
   :param clusters: the selected Clustering algo
   :param similar_events_x_parameter: the selected x-value for 
      configuring the similar events
   :param similar_events_y_parameter: the selected y-value for 
      configuring the similar events
   :return: the dynamic page of similar events that shows the 
      Tabulator object of the similar events identified

|

.. py:function:: web_dashboard.dynamic_env.build_similar_event_page._download_callback():
   
   The method is a callback activated by the download button.

   :return: the CSV file of the summary of the selected events

|

.. py:function:: web_dashboard.dynamic_env.build_similar_event_page.build_similar_events(_):
   
   This method returns the dynamic event page(s) for the selected similar
   event(s).
   It depends on the similar events selected by the user on the 
   Tabulator object, and the clicking of the display button.

   :return: the dynamic event page(s) that show(s) the event data, and
      the event waveforms of the selected similar event(s)

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
