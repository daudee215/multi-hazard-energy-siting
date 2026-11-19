"""Shared project figure theme; edit ../figure_style.json to restyle consistently."""
from pathlib import Path
import json
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
STYLE=json.loads((Path(__file__).resolve().parents[1]/'figure_style.json').read_text())
FONT=STYLE['font'];COLORS=STYLE['colors'];INK=STYLE['ink']
# Register the installed faces explicitly so a stale Matplotlib font cache cannot substitute a font.
for suffix in ['Regular','Bold','Italic','BoldItalic']:
 path=Path('C:/Windows/Fonts')/(FONT.replace(' ','')+'-'+suffix+'.ttf')
 if path.exists():font_manager.fontManager.addfont(str(path))
FONT_PATH=font_manager.findfont(FONT,fallback_to_default=False)
SEQUENTIAL=LinearSegmentedColormap.from_list('project_sequential',STYLE['sequential'])
UNCERTAINTY=LinearSegmentedColormap.from_list('project_uncertainty',STYLE['uncertainty'])
mpl.rcParams.update({'font.family':FONT,'font.size':10,'axes.titlesize':13,'axes.labelsize':10,
 'axes.titleweight':'bold','axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
 'xtick.labelsize':9,'ytick.labelsize':9,'axes.edgecolor':'#ACB1B4','axes.linewidth':.7,
 'axes.facecolor':'white','figure.facecolor':'white','savefig.facecolor':'white',
 'axes.prop_cycle':cycler(color=COLORS),'lines.linewidth':1.8,'legend.frameon':False,
 'pdf.fonttype':42,'ps.fonttype':42})
