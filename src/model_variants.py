from src.models.fastkan import get_fastkan_model
from src.models.fourierkan import get_fourierkan_model
from src.models.efficient_kan import get_efficientkan_model
from src.models.wavkan import get_wavkan_model
from src.models.oae import get_orthogonalAE_model
# optionally other models too:
# from src.models.fourier_kan import get_fourierkan_model

MODEL_VARIANTS = {
    "FastKAN": get_fastkan_model,
    "FourierKAN": get_fourierkan_model,
    "EfficientKAN": get_efficientkan_model,
    "WavKAN": get_wavkan_model,
    "OrthogonalAE": get_orthogonalAE_model

}