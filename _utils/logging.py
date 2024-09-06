import wandb as wdb
from PIL import Image
import io

def fig_to_pil(fig):
    #matplotlib figure to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return wdb.Image(Image.open(buf))