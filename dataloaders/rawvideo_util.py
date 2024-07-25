import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2
import os
import json
import gzip
import logging

class RawVideoExtractorCV2():

    _CACHE_DIR = 'extractorCache'
    _CACHE_INDEX_FN = 'cacheIndex.json'

    def __init__(self, centercrop=False, size=224, framerate=-1, use_caching:bool = True, logger=None):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
        self.use_caching = use_caching
        if logger is None:
            logger = logging
        self.logger = logger

        if (self.use_caching):
            self.cacheDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RawVideoExtractorCV2._CACHE_DIR)
            if not os.path.exists(self.cacheDir):
                os.mkdir(self.cacheDir)
            self.cacheIndexFn = os.path.join(self.cacheDir, RawVideoExtractorCV2._CACHE_INDEX_FN)
            if not os.path.exists(self.cacheIndexFn):
                self.cacheIndex = {'next_index_to_use':0, 'cache_map':{}}
                self.saveUpdatedCacheIndex()
            else:
                with open(self.cacheIndexFn) as f:
                    self.cacheIndex = json.load(f)

    def refreshCache(self):
        with open(self.cacheIndexFn) as f:
            self.cacheIndex = json.load(f)

    def saveUpdatedCacheIndex(self):
        self.logger.debug(f"updating cache index")
        with open(self.cacheIndexFn, 'w') as f:
            json.dump(self.cacheIndex, f, indent=4)
        self.logger.debug(f"cacheIndex now has {len(self.cacheIndex['cache_map'])} entries")
    
    def saveDataAndUpdateCache(self, image_input, video_path, start_time, end_time):
        fn = f"{self.cacheIndex['next_index_to_use']}"
        self.cacheIndex['next_index_to_use'] += 1
        self.logger.debug(f"saving {video_path} to entry {fn} of the cache")
        with gzip.GzipFile(os.path.join(self.cacheDir, fn), "x") as f:
            np.save(f, image_input['video'].numpy())
        self.cacheIndex['cache_map'][self.getCacheIndexName(video_path, start_time, end_time)] = fn
        self.saveUpdatedCacheIndex()

    def getCacheIndexName(self, video_path, start_time, end_time):
        return f"{video_path}_{start_time}_{end_time}_{self.centercrop}_{self.size}_{self.framerate}" 

    def isCached(self, video_path, start_time, end_time):
        cacheIdxName = self.getCacheIndexName(video_path, start_time, end_time)
        retVal = cacheIdxName in self.cacheIndex['cache_map'].keys()
        if retVal: 
            assert os.path.exists(os.path.join(self.cacheDir,self.cacheIndex['cache_map'][cacheIdxName])), f"key found in the index but corresponding data is not available"
        return retVal

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None):
        if self.use_caching and self.isCached(video_path, start_time, end_time):
            self.logger.debug(f"cache hit: {video_path}")
            image_input_fn =  self.cacheIndex['cache_map'][self.getCacheIndexName(video_path, start_time, end_time)]
            with gzip.GzipFile(os.path.join(self.cacheDir,image_input_fn), 'r') as f:
                image_input = {'video':th.tensor(np.load(f))}
            return image_input
        else:
            if self.use_caching:
                self.logger.info(f"cache miss: {video_path}")
            image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
            if self.use_caching:
                self.saveDataAndUpdateCache(image_input, video_path, start_time, end_time)
            return image_input

    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2