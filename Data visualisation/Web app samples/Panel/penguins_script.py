import panel as pn
import numpy as np
import pandas as pd
import holoviews as hv
import hvplot.pandas # noqa

hv.extension('bokeh')
pn.extension()

penguins = pd.read_csv('penguins.csv')

colors = {
    'Adelie Penguin': '#1f77b4',
    'Gentoo penguin': '#ff7f0e',
    'Chinstrap penguin': '#2ca02c'
}

scatter = penguins.hvplot.scatter('Culmen Length (mm)',
 'Culmen Depth (mm)', c='Species', cmap=colors,
  responsive=True, min_height=250, max_height=500,
   min_width=250, title="test")
histogram = penguins.hvplot.hist('Body Mass (g)', by='Species',
 color=hv.dim('Species').categorize(colors), legend=False,
  alpha=0.5, responsive=True, min_height=300, title="test")

welcome = "## Welcome and meet the Palmer penguins!"
instructions = """
### Instructions

Use the box-select and lasso-select tools to select a subset of penguins
and reveal more information about the selected subgroup through the power
of cross-filtering.
"""
license = """
### License

Data are available by CC-0 license in accordance with the Palmer Station LTER Data Policy and the LTER Data Access Policy for Type I data."
"""
art = pn.Column(welcome, instructions, license, sizing_mode="stretch_width")

pn.config.raw_css.append("body .bk-root { font-family: Ubuntu !important; }")
material = pn.template.MaterialTemplate(title='Palmer Penguins')

header = pn.Row(
    pn.layout.HSpacer(),
    pn.Spacer(width=100),
    sizing_mode='stretch_width'
)

material.header.append(header)
material.main.append(art)
material.main.append(pn.Row(
    pn.Column(scatter, scatter, sizing_mode="stretch_width"), 
    scatter, sizing_mode="stretch_width"))
material.main.append(pn.Row(histogram, 
    sizing_mode="stretch_width"))

material.servable()
