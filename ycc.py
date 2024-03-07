from fastbook import DataBlock, ImageBlock, CategoryBlock, RandomSplitter, parent_label, Resize, DataLoaders, DataLoader, PILImage
from fastcore.all import L
from fastai.vision.all import Image, resize_images, verify_images, get_image_files, cnn_learner, resnet18, resnet50, error_rate, ClassificationInterpretation, accuracy
from fastai.vision.learner import vision_learner
from fastai.vision.widgets import ImageClassifierCleaner, shutil
from fastai.callback.schedule import fine_tune
from pathlib import Path
from time import sleep

path = Path('E:/gDrive/38.Pic/train/t1')
# path = Path('E:/models/.fastai/data/cat')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize((540, 360), method='squish')]
).dataloaders(path,bs=14)
# dls.show_batch(max_n=30)
learn = vision_learner(dls, resnet50, metrics=error_rate)
learn.fine_tune(5)
