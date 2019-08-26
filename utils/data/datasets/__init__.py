from .basement import Dataset, ImageDataset, VideoDataset
from .basement import register_image_dataset
from .basement import register_video_dataset
from .datamanager import ImageDataManager, VideoDataManager


from . import cuhk02
from . import market1501

__Factory__ = {
	"cuhk02": cuhk02,
	"market1501": market1501,
}

