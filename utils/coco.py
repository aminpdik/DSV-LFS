r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST3,COCOclasses
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
import random
from model.llava import conversation as conversation_lib

class DatasetCOCO(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(self, datapath,samples_per_epoch,image_size,vision_tower, fold, split, shot):

        self.samples_per_epoch=samples_per_epoch
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        # self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = datapath

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
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
        return self.samples_per_epoch if self.split == 'trn' else 2000


    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, sampled_classes,org_qry_imsize = self.load_frame()
        query_name=query_name.split('/')[1].replace('.jpg', '')
        support_names1=[s.split('/')[1].replace('.jpg', '') for s in support_names]
        support_names=support_names1
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
        # cv2.waitKey(0)
        resize = Qimage.shape[:2]
        Sresize = Simage.shape[:2]


        query_mask = query_mask.float()
        # if not self.use_original_imgsize:
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

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        if self.fold == 4:
            class_ids_val=[-1]
        else:
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        #class_ids_val=[-1]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        if self.fold == 4:
            class_ids_val=class_ids_trn
        else:
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]


        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        sampled_classes = [COCOclasses[class_sample + 1]]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]

        query_img = cv2.imread(os.path.join(self.base_path, query_name))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:

            support_img = cv2.imread(os.path.join(self.base_path, support_name))
            support_img = cv2.cvtColor(support_img, cv2.COLOR_BGR2RGB)
            support_imgs.append(support_img)

            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample,sampled_classes, org_qry_imsize

