r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST3,COCOclasses
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
import random
import cv2

class DatasetPASCAL(torch.utils.data.Dataset):

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, datapath,samples_per_epoch,image_size,vision_tower, fold, split, shot):

        self.samples_per_epoch=samples_per_epoch
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot

        self.img_path = os.path.join(datapath, 'JPEGImages/')
        self.ann_path = os.path.join(datapath, 'SegmentationClassAug/')
        #print('TEST2')
        #self.testepisodes=open('/home/ampdi/scratch/FInal_WACV/LISA-FEWSHOT/test_fold1_pascal/Test2.txt','r').readlines()
        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.short_question_list = SHORT_QUESTION_LIST3
        self.answer_list = ANSWER_LIST
        self.ClassPrompts = {}
        self.PascalClases=open("./ClassPrompt/PASCAL/pascalC.txt",'r').readlines()

        for Cname in self.PascalClases:
            if Cname[-1]=='\n':
                Cname=Cname[:-1]
            # Cname = COCOclasses[C]
            # print(Cname)
            f = open("./ClassPrompt/PASCAL/" + Cname + ".txt", "r").readline()
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

        if self.split == 'trn':
            return self.samples_per_epoch
        else:
            return 2000

    def __getitem__(self, idx):

        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        #print(query_name)
        sampled_classes=[self.PascalClases[class_sample]]
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)
        query_cmask = self.extract_ignore_idx(query_cmask, class_sample)
        support_cmasks1 = [self.extract_ignore_idx(S, class_sample) for S in support_cmasks]
        support_cmasks=support_cmasks1


        # preprocess image for clip
        Qimage_clip = self.clip_image_processor.preprocess(
            query_img, return_tensors="pt"
        )["pixel_values"][0]

        # ORGresize=query_img.shape[:2]
        ORGresize=(400,400)
        Qimage,QmaskSAM = self.transform.apply_image(query_img,query_cmask)  # preprocess image for sam
        # QmaskSAM=QmaskSAM.astype(np.float32)
        # QmaskSAM=self.extract_ignore_idx(QmaskSAM, class_sample)

        Simage,SmaskSAM = self.transform.apply_image(support_imgs[0],support_cmasks[0])  # preprocess image for sam
        # SmaskSAM = SmaskSAM.astype(np.float32)
        # SmaskSAM = self.extract_ignore_idx(SmaskSAM, class_sample)


        # cv2.imshow('Qim',Qimage)
        # cv2.imshow('Qmask',255*QmaskSAM)
        # cv2.imshow('Sim', Simage)
        # cv2.imshow('Smask', 255 * SmaskSAM)
        # cv2.waitKey(0)
        resize = Qimage.shape[:2]
        Sresize = Simage.shape[:2]



        # if not self.use_original_imgsize:
        query_mask = F.interpolate(query_cmask.unsqueeze(0).unsqueeze(0).float(), ORGresize,
                                   mode='nearest').squeeze()


        support_imgs = torch.stack([self.clip_image_processor.preprocess(
            support_img, return_tensors="pt"
        )["pixel_values"][0] for support_img in support_imgs])


        for midx, smask in enumerate(support_cmasks):
            support_cmasks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:],
                                                mode='nearest').squeeze()

        support_masks = torch.stack(support_cmasks)


        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            if text[-1]=='\n':
                text=text[:-1]

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

    def extract_ignore_idx(self, mask, class_id):
        # boundary = (mask / 255).floor()
        mask[mask>25]=0
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        img = cv2.imread(os.path.join(self.img_path, img_name) + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]
        
          
        #seed = int(random.random() * 1e5)
        #np.random.seed(seed)

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        if self.fold == 4:
            class_ids_val=[-1]
        else:
            class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]


        #class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        #class_ids_val=[-1]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        
        if self.fold == 4:
            class_ids_val=class_ids_trn
        else:
            class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]


        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('./splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            if self.fold == 4:

                for fold_id in range(self.nfolds):
                    if fold_id == self.fold:  # Skip validation fold
                        continue
                    img_metadata += read_metadata(self.split, fold_id)

            else:

                img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
