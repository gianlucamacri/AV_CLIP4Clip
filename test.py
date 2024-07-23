from torchinfo import summary
from main_task_retrieval import init_model, init_device, set_seed_logger
import logging
import torch
import numpy as np
from dataloaders.data_dataloaders import dataloader_artistic_videos_test, dataloader_artistic_videos_train
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

global logger

class Args:
    def __init__(self):
        self.do_pretrain = False
        self.do_train = True
        self.do_eval = False
        self.test_best_model = True
        self.train_csv = None
        self.val_csv = None
        self.data_path = None
        self.features_path = None
        self.metadata_fn = "metadata_no_accents"
        self.video_dir = "compressedVideos"
        self.split = "75_5_10_split_3strat_cap_75"
        self.merge_test_val = False
        self.best_model_strategy = "loss"
        self.use_caching = True
        self.num_thread_reader = 1
        self.lr = 5e-05
        self.epochs = 10
        self.batch_size = 8
        self.batch_size_val = 8
        self.lr_decay = 0.9
        self.n_display = 50
        self.seed = 42
        self.max_words = 20
        self.max_frames = 12
        self.feature_framerate = 1
        self.margin = 0.1
        self.hard_negative_rate = 0.5
        self.negative_weighting = 1
        self.n_pair = 1
        self.output_dir = "/media/gian/Volume/Data/ckpts/av_retreival_lr5e-05_e10_b8_12fps_loss_42_1721493284"
        self.cross_model = "cross-base"
        self.init_model = None
        self.resume_model = None
        self.do_lower_case = False
        self.warmup_proportion = 0.1
        self.gradient_accumulation_steps = 1
        self.n_gpu = 1
        self.cache_dir = ""
        self.fp16 = False
        self.fp16_opt_level = "O1"
        self.task_type = "retrieval"
        self.datatype = "artistic_videos"
        self.world_size = 1
        self.local_rank = 0
        self.rank = 0
        self.coef_lr = 0.001
        self.use_mil = False
        self.sampled_use_mil = False
        self.text_num_hidden_layers = 12
        self.visual_num_hidden_layers = 12
        self.cross_num_hidden_layers = 4
        self.loose_type = True
        self.expand_msrvtt_sentences = False
        self.train_frame_order = 0
        self.eval_frame_order = 0
        self.freeze_layer_num = 0
        self.slice_framepos = 2
        self.linear_patch = "2d"
        self.sim_header = "meanP"
        self.pretrained_clip_name = "ViT-B/16"
        self.load_model_from_fn = None



def main():
    args = Args()
    args = set_seed_logger(args)
    tokenizer = ClipTokenizer()
    test_dataloader, n = dataloader_artistic_videos_test(args, tokenizer, subset='test')

    device, n_gpu = init_device(args, args.local_rank)
    print(f"device: {device}")
    model = init_model(args, device, n_gpu, args.local_rank)
    # for batch in test_dataloader:

    #     #batch = tuple(t.to(device=device, non_blocking=True) for t in batch)


    #     break
    with open(f'{args.pretrained_clip_name.replace("/", "_")}.txt', 'w') as f:
        f.write(str(model))


if __name__=='__main__':
    main()