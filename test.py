from bokeh.palettes import Spectral11
from bokeh.models import CategoricalColorMapper
print(Spectral11)


color_mapping = CategoricalColorMapper(factors=[str(l) for l in  [1, 2, 3, 10]], palette=Spectral11)
print(color_mapping)
