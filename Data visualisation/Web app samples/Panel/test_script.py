import panel as pn
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas # noqa
from bokeh.resources import INLINE

import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

hv.extension('bokeh')
pn.extension()

df = pd.read_csv("test_cnnoutputs.csv", header=0)

scatter_voltage = df.hvplot.scatter(
    x='Vab_sag',
    y='Vab_swell',
    by='events',
    legend=True,
    responsive=True,
    height=700,
    width=600,
    title="Scatter-plot")

vab_wave = df.hvplot.line(
    x='timestamps (in ms)',
    y='Vab',
    legend=False,
    responsive=True,
    height=100,
    width=400,
    title="Waveform of Voltage A to B",
    line_width=1
)

vbc_wave = df.hvplot.line(
    x='timestamps (in ms)',
    y='Vbc',
    legend=False,
    responsive=True,
    height=100,
    width=400,
    title="Waveform of Voltage B to C",
    line_width=1
)

vca_wave = df.hvplot.line(
    x='timestamps (in ms)',
    y='Vca',
    legend=False,
    responsive=True,
    height=100,
    width=400,
    title="Waveform of Voltage C to A",
    line_width=1
)

ia_wave = df.hvplot.line(
    x='timestamps (in ms)',
    y='Ia',
    legend=False,
    responsive=True,
    height=100,
    width=400,
    title="Waveform of Current A",
    line_width=1
)

ib_wave = df.hvplot.line(
    x='timestamps (in ms)',
    y='Ib',
    legend=False,
    responsive=True,
    height=100,
    width=400,
    title="Waveform of Current B",
    line_width=1
)

ic_wave = df.hvplot.line(
    x='timestamps (in ms)',
    y='Ic',
    legend=False,
    responsive=True,
    height=100,
    width=400,
    title="Waveform of Current C",
    line_width=1
)

fontsize_value = '60%'
vab_wave.opts(fontsize={
    'title': fontsize_value,
    'labels': fontsize_value, 
    'ticks': fontsize_value, 
})
vbc_wave.opts(fontsize={
    'title': fontsize_value,
    'labels': fontsize_value, 
    'ticks': fontsize_value, 
})
vca_wave.opts(fontsize={
    'title': fontsize_value,
    'labels': fontsize_value, 
    'ticks': fontsize_value, 
})
ia_wave.opts(fontsize={
    'title': fontsize_value,
    'labels': fontsize_value, 
    'ticks': fontsize_value, 
})
ib_wave.opts(fontsize={
    'title': fontsize_value,
    'labels': fontsize_value, 
    'ticks': fontsize_value, 
})
ic_wave.opts(fontsize={
    'title': fontsize_value,
    'labels': fontsize_value, 
    'ticks': fontsize_value, 
})

similar_events = pd.DataFrame({
    "Similar events": ["event_001", "event_002", "event_003",
    "event_004", "event_005", "event_006", 
    "event_007", "event_008", "event_009", 
    "event_010", "event_011", "event_012"],
})
similar_events_df = similar_events.style.set_properties(**{'text-align': 'left'})
similar_events = pn.pane.DataFrame(
    similar_events_df, 
    escape=False, 
    width=600
)

pn.config.raw_css.append("body .bk-root { font-family: Ubuntu !important; }")
material = pn.template.MaterialTemplate(title='Smart Power Grids Dashboard')

header = pn.Row(
    pn.layout.HSpacer(),
    pn.Spacer(width=100)
)

welcome = "## Welcome to the Smart Power Grids Dashboard"
instructions = """
### Instructions

Step 1. User to upload the CNN outputs (perhaps/ likely in CSV format) to the web app<br>
Step 2. User to choose which clustering algo to run on the datasets, and perhaps the two variables for plotting the graph<br>
Step 3. The web app would plot the data in a scatter-plot<br>
Step 4. When we select a dot on the plot, that dot represents an event. The web app would show the following:
<ul>
<li>The event's 6 waveforms (3 voltages, 3 currents)
<li>Similar events to that one
<li>The 6 waveforms of one of the similar events
</ul>
Final step. The user should be able to perform labelling on the plot<br>
"""
art = pn.Column(
    welcome, 
    instructions, 
    sizing_mode="stretch_width"
)
file_input = pn.widgets.FileInput(accept='.csv', multiple=True)

material.header.append(header)
material.main.append(art)
material.main.append(
    pn.Row(
        "### Please upload your CNN outputs here.<br>"
    )
)
material.main.append(
    file_input
)
material.main.append(
    pn.Spacer(height=50)
)
material.main.append(
    pn.Row(
        scatter_voltage, 
        pn.Column(
            vab_wave,
            vbc_wave,
            vca_wave,
            ia_wave,
            ib_wave,
            ic_wave
        )
    )
)
material.main.append(
    pn.Spacer(height=50)
)
material.main.append(
    pn.Row(
        similar_events,
        pn.Column(
            vab_wave,
            vbc_wave,
            vca_wave,
            ia_wave,
            ib_wave,
            ic_wave
        )
    )
)

material.servable()
material.save('test_scripts.html', resources=INLINE)
