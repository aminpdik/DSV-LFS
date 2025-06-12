
import os
import shutil
import sys
import time
from functools import partial
from utils.config import parse_args,GenDSConfig
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.DSVLFS import DSVLFSForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, dict_to_cuda,
                         build_img_metadata,PrepResume)
import random
from utils.dataset import FSSDataset
from utils.logger import  AverageMeter1
from utils.evaluation import Evaluator


def CreateModelTokenizer(args):

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEM]", special_tokens=True)
    args.sem_token_idx = tokenizer("[SEM]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "sem_token_idx": args.sem_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = DSVLFSForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    # if not args.eval_only:
    model.get_model().initialize_DSVLFS_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                        isinstance(module, cls)
                        and all(
                    [
                        x not in name
                        for x in [
                        "visual_model",
                        "vision_tower",
                        "mm_projector",
                        "text_hidden_fcs",
                        "encoder_layerFewshot",
                        "decoderFewshot",
                    ]
                    ]
                )
                        and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "encoder_layerFewshot",
                              "decoderFewshot"]
                ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True


    return model,tokenizer


def main(args):


    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    ## Define model and tokenizer
    model,tokenizer=CreateModelTokenizer(args)


    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # create train and val dataset
    if args.benchmark=='pascal':
        AllTrnIms=build_img_metadata('trn', 4, args.fold)
        args.steps_per_epoch=int(len(AllTrnIms)/(args.batch_size*args.grad_accumulation_steps*world_size))

    #print(args.steps_per_epoch)
    samples_per_epoch = args.batch_size*args.grad_accumulation_steps*args.steps_per_epoch*world_size
    # print('-----',samples_per_epoch)

    FSSDataset.initialize(datapath=args.dataset_dir)
    train_dataset = FSSDataset.build_dataset(args.benchmark,samples_per_epoch,args.image_size,args.vision_tower, args.fold, 'trn')


    if args.no_eval == False:
        val_dataset = FSSDataset.build_dataset(args.benchmark, samples_per_epoch, args.image_size, args.vision_tower,
                                               args.fold, 'val')
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")


    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=GenDSConfig(args),
    )

    args, model_engine=PrepResume(args, model_engine)

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        miou, fbiou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    print('train class ids= ', train_dataset.class_ids)
    print('val class ids= ', val_dataset.class_ids)
    print('fold= ', args.fold)
    
    for epoch in range(args.start_epoch, args.epochs):
        # miou, fbiou = validate(val_loader, model_engine, epoch, writer, args)
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            miou, fbiou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = miou > best_score
            best_score = max(miou, best_score)
            cur_fbiou = fbiou if is_best else cur_fbiou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_miou{:.3f}_fbiou{:.3f}.pth".format(
                            best_score, cur_fbiou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)







def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(
        train_loader,
        model,
        epoch,
        scheduler,
        writer,
        train_iter,
        args,
):

    fix_randseed(None)

    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    # mask_FewShot_losses = AverageMeter("MaskFShotLoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            # mask_FewShot_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                input_dict["Simage_clip"] = input_dict["Simage_clip"].bfloat16()
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["Simages"] = input_dict["Simages"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            # FewShotLoss = output_dict["FewShotLoss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            # mask_FewShot_losses.update(FewShotLoss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                # mask_FewShot_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                # writer.add_scalar(
                #     "train/FewShotLoss", mask_FewShot_losses.avg, global_step
                # )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            # mask_FewShot_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):

    fix_randseed(0)
    average_meter1 = AverageMeter1(val_loader.dataset)

    model_engine.eval()

    idx = -1
    for input_dict in tqdm.tqdm(val_loader):
        idx += 1
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            input_dict["Simage_clip"] = input_dict["Simage_clip"].bfloat16()
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["Simages"] = input_dict["Simages"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"].int()
            output_list = (pred_masks[0] > 0).int()

        area_inter, area_union = Evaluator.classify_prediction(output_list, masks_list, None)
        average_meter1.update(area_inter, area_union, input_dict["class_sample_list"], None)
        average_meter1.write_process(idx, len(val_loader), epoch, write_batch_idx=50)

    miouFewshot, fb_iouFewshot = average_meter1.compute_iou()

    if args.local_rank == 0:
        print("miou: {:.4f}, fbiou: {:.4f}".format(miouFewshot, fb_iouFewshot))

    return miouFewshot, fb_iouFewshot


if __name__ == "__main__":
    main(sys.argv[1:])

