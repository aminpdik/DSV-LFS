r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.pascal import DatasetPASCAL
from utils.coco import DatasetCOCO
from utils.fss import DatasetFSS
import torch
from model.llava.mm_utils import tokenizer_image_token
from model.llava import conversation as conversation_lib
from .utils import (IGNORE_INDEX,DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):

    Qimage_clip_list=[]
    Qimage_list=[]
    QmaskSAM_list=[]
    Simage_list=[]
    SmaskSAM_list=[]
    query_mask_list=[]
    support_imgs_list=[]
    support_masks_list=[]

    conversation_list=[]
    resize_list=[]
    ORGresize_list=[]
    questions_list=[]

    query_name_list=[]
    support_names_list=[]
    class_sample_list=[]
    sampled_classes_list=[]

    offset_list = [0]
    cnt = 0
    inferences = []
    for (
            Qimage_clip,
            Qimage,
            QmaskSAM,
            Simage,
            SmaskSAM,
            query_mask,
            support_imgs,
            support_masks,

            conversations,
            resize,
            ORGresize,
            questions,

            query_name,
            support_names,
            class_sample,
            sampled_classes,
            inference

    ) in batch:

        Qimage_clip_list.append(Qimage_clip),

        Qimage_list.append(Qimage),
        Simage_list.append(Simage),
        QmaskSAM_list.append(QmaskSAM),
        SmaskSAM_list.append(SmaskSAM),

        query_mask_list.append(query_mask),
        support_imgs_list.append(support_imgs),
        support_masks_list.append(support_masks),

        conversation_list.extend(conversations),
        resize_list.append(resize),
        ORGresize_list.append(ORGresize)
        questions_list.append(questions),

        query_name_list.append(query_name),
        support_names_list.append(support_names),
        class_sample_list.append(class_sample),
        sampled_classes_list.append(sampled_classes),
        inferences.append(inference)
        cnt += len(conversations)
        offset_list.append(cnt)

    class_sample_list=torch.stack([torch.tensor(x) for x in class_sample_list])
    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )


    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {

        "images_clip": torch.stack(Qimage_clip_list, dim=0),
        "images": torch.stack(Qimage_list, dim=0),

        "Simages":torch.stack(Simage_list, dim=0),
        "QmaskSam":torch.stack(QmaskSAM_list, dim=0),
        "SmaskSam": torch.stack(SmaskSAM_list, dim=0),

        "masks_list": torch.stack(query_mask_list),

        "Simage_clip": torch.stack(support_imgs_list, dim=0),
        "support_masks_list":  torch.stack(support_masks_list, dim=0),

        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,


        "resize_list": resize_list,
        "ORGresize_list":ORGresize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "class_sample_list":class_sample_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "query_name_list":query_name,
        "support_names_list":support_names,
    }




class FSSDataset:

    @classmethod
    def initialize(cls, datapath):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        cls.datapath = datapath

    @classmethod
    def build_dataset(cls, benchmark, samples_per_epoch,image_size,vision_tower, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility

        dataset = cls.datasets[benchmark](cls.datapath,samples_per_epoch=samples_per_epoch,image_size=image_size,vision_tower=vision_tower, fold=fold, split=split, shot=shot)

        return dataset
