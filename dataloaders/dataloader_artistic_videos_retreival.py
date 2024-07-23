from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from dataloaders.rawvideo_util import RawVideoExtractor
import logging
from datasets.artistic_video_dataset.utils import ArtisticVideoDataset

### adapted from MSR-VTT
# edited the deprecated numpy types according to https://numpy.org/devdocs/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated 

class Artistic_Videos_DataLoader(Dataset):

    _CLIP_TOKEN_LIMIT = 77

    """artistic videos dataset loader."""
    def __init__(
            self,
            metadata_fn, # metadata (old csvpath)
            video_dir, # video path, i'll use this over the video path in order to get a better generalization (e.g. use compressed videos)
            split_fn,
            split,
            merge_test_val,
            tokenizer,
            max_tokens:int=_CLIP_TOKEN_LIMIT,
            video_framerate:int=1,
            max_frames:int=100,
            image_resolution:int=224,
            frame_order:int=0,
            slice_framepos:int=0,
            use_caching=True,
    ):
        
        if not 'logger' in globals():
            logger = logging
        self.logger = logger
        avd = ArtisticVideoDataset(metadata_fn, video_dir)
        if not merge_test_val:
            data_df = avd.getMetadata(split_fn=split_fn, split=split).reset_index(inplace=False) # reset may be unnecessary
        else:
            self.logger.info('merging test and validation split')
            test_df = avd.getMetadata(split_fn=split_fn, split='test').reset_index(inplace=False) # reset may be unnecessary
            val_df = avd.getMetadata(split_fn=split_fn, split='validation').reset_index(inplace=False) # reset may be unnecessary
            data_df = pd.concat(test_df, val_df).reset_index(inplace=False)
        self.data = data_df # .iloc[:10] # this is a shortcut to persof quick testing on smaller datasets

        self.video_framerate = video_framerate
        self.max_tokens = max_tokens
        if max_tokens > Artistic_Videos_DataLoader._CLIP_TOKEN_LIMIT:
            self.logger.warning(f'provided max_tokens value of {max_tokens} is above the clip limit, capping it to {Artistic_Videos_DataLoader._CLIP_TOKEN_LIMIT}')
            self.max_tokens = Artistic_Videos_DataLoader._CLIP_TOKEN_LIMIT

        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=video_framerate, size=image_resolution, use_caching=use_caching)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
    def refreshCache(self):
        self.rawVideoExtractor.refreshCache()

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):

        pairs_text = np.zeros((1, self.max_tokens), dtype=int)
        pairs_mask = np.zeros((1, self.max_tokens), dtype=int)
        pairs_segment = np.zeros((1, self.max_tokens), dtype=int)

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + self.tokenizer.tokenize(sentence)
        total_length_with_CLS = self.max_tokens - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_tokens:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_tokens
        assert len(input_mask) == self.max_tokens
        assert len(segment_ids) == self.max_tokens

        pairs_text[0] = np.array(input_ids)
        pairs_mask[0] = np.array(input_mask)
        pairs_segment[0] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, [video_id]

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        for i, video_path in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            assert video_path[-4:].lower() == ".mp4", f"file video should end with '.mp4' extension, found: {video_path}"
            if os.path.exists(video_path) is False:
                raise Exception(f"cannot find video element at {video_path}")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_path))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['filename'].values[idx]
        sentence = self.data['description'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask
    
    def getVideoIds(self):
        return self.data['video_id'].values.tolist()
