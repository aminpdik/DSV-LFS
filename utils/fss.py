r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST3,COCOclasses
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
import cv2
import random
from model.llava import conversation as conversation_lib

class DatasetFSS(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(self, datapath,samples_per_epoch,image_size,vision_tower, fold, split, shot):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open('./splits/fss/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.short_question_list = SHORT_QUESTION_LIST3
        self.answer_list = ANSWER_LIST

        self.ClassPrompts = {}
        for C in COCOclasses:
            Cname = COCOclasses[C]
            # print(Cname)
            f = open("./ClassPrompt/" + Cname + ".txt", "r").readline()
            self.ClassPrompts[Cname] = f

    def preprocess(self, x: torch.Tensor,mask) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        mask = F.pad(mask, (0, padw, 0, padh))


        return x,mask

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):

        query_name, support_names, class_sample = self.sample_episode(idx)
        indxclassName=len(query_name.split('/'))-2
        sampled_classes=[query_name.split('/')[indxclassName]]
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        # preprocess image for clip
        Qimage_clip = self.clip_image_processor.preprocess(
            query_img, return_tensors="pt"
        )["pixel_values"][0]

        # ORGresize=query_img.shape[:2]
        ORGresize=(400,400)
        Qimage,QmaskSAM = self.transform.apply_image(query_img,query_mask)  # preprocess image for sam
        Simage,SmaskSAM = self.transform.apply_image(support_imgs[0],support_masks[0])  # preprocess image for sam

        # cv2.imshow('Qim',Qimage)
        # cv2.imshow('Qmask',255*QmaskSAM)
        #
        # cv2.imshow('Sim', Simage)
        # cv2.imshow('Smask', 255 * SmaskSAM)
        #
        #
        # cv2.waitKey(0)
        resize = Qimage.shape[:2]
        Sresize = Simage.shape[:2]



        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), ORGresize,
                                   mode='nearest').squeeze()


        support_imgs = torch.stack([self.clip_image_processor.preprocess(
            support_img, return_tensors="pt"
        )["pixel_values"][0] for support_img in support_imgs])


        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                                mode='nearest').squeeze()

        support_masks = torch.stack(support_masks)


        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            question_template=question_template.replace('{class_name}',text.lower())
            question_template = question_template.replace('{prop}', self.ClassPrompts[text])
            questions.append(question_template)

            answertemp=random.choice(self.answer_list)
            # answertemp = answertemp.replace('{class_name}', text.lower())
            answers.append(answertemp)


        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        Qimage,QmaskSAM = self.preprocess(torch.from_numpy(Qimage).permute(2, 0, 1).contiguous(),torch.from_numpy(QmaskSAM).contiguous())
        Simage,SmaskSAM = self.preprocess(torch.from_numpy(Simage).permute(2, 0, 1).contiguous(),torch.from_numpy(SmaskSAM).contiguous())


        if self.split=='trn':
            inference=False
        else:
            inference = True

        return Qimage_clip,Qimage,QmaskSAM,Simage,SmaskSAM,\
            query_mask,\
            support_imgs[0],\
            support_masks[0],\
            conversations,\
            resize, \
            ORGresize,\
            questions,\
            query_name,\
            support_names,\
            class_sample,\
            sampled_classes, \
            inference

    def load_frame(self, query_name, support_names):

        query_img = cv2.imread(query_name)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        support_imgs = [cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB) for name in support_names]


        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)
        return img_metadata
