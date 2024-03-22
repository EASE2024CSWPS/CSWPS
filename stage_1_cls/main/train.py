import argparse
import json
import logging
import os
import subprocess
import sys

import c2nl.config as config
import c2nl.inputters.dataset as data
import c2nl.inputters.utils as util
import c2nl.inputters.vector as vector
import numpy as np
import torch
from c2nl.inputters import constants
from c2nl.inputters.timer import AverageMeter, Timer
from main.model import CodClassifier
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

logger = logging.getLogger()
ml_logger = None


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."),
        ["", "K", "M", "B", "T"][magnitude],
    )


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register("type", "bool", str2bool)

    # Runtime environment
    runtime = parser.add_argument_group("Environment")
    runtime.add_argument(
        "--data_workers",
        type=int,
        default=5,
        help="Number of subprocesses for data loading",
    )
    runtime.add_argument(
        "--random_seed",
        type=int,
        default=1013,
        help=(
            "Random seed for all numpy/torch/cuda "
            "operations (for reproducibility)"
        ),
    )
    runtime.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="Train data iterations",
    )
    runtime.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    runtime.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="Batch size during validation/testing",
    )

    # Files
    files = parser.add_argument_group("Filesystem")
    files.add_argument(
        "--dataset_name",
        nargs="+",
        type=str,
        required=True,
        help="Name of the experimental dataset",
    )
    files.add_argument(
        "--model_dir",
        type=str,
        default="/tmp/qa_models/",
        help="Directory for saved models/checkpoints/logs",
    )
    files.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Unique model identifier (.mdl, .txt, .checkpoint)",
    )
    files.add_argument(
        "--data_dir",
        type=str,
        default="/data/",
        help="Directory of training/validation data",
    )
    files.add_argument(
        "--train_src",
        nargs="+",
        type=str,
        help="Preprocessed train source file",
    )
    files.add_argument(
        "--train_repo",
        nargs="+",
        type=str,
        help="Preprocessed train repo file",
    )
    files.add_argument(
        "--dev_src",
        nargs="+",
        type=str,
        required=True,
        help="Preprocessed dev source file",
    )
    files.add_argument(
        "--dev_repo",
        nargs="+",
        type=str,
        help="Preprocessed dev repo file",
    )

    # Saving + loading
    save_load = parser.add_argument_group("Saving/Loading")
    save_load.add_argument(
        "--checkpoint",
        type="bool",
        default=False,
        help="Save model + optimizer state after each epoch",
    )
    save_load.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to a pretrained model to warm-start with",
    )

    # Data preprocessing
    preprocess = parser.add_argument_group("Preprocessing")
    preprocess.add_argument(
        "--max_examples",
        type=int,
        default=-1,
        help="Maximum number of examples for training",
    )
    preprocess.add_argument(
        "--uncase",
        type="bool",
        default=False,
        help="Code and summary words will be lower-cased",
    )
    preprocess.add_argument(
        "--src_vocab_size",
        type=int,
        default=None,
        help="Maximum allowed length for src dictionary",
    )
    preprocess.add_argument(
        "--max_characters_per_token",
        type=int,
        default=30,
        help="Maximum number of characters allowed per token",
    )

    # General
    general = parser.add_argument_group("General")
    general.add_argument(
        "--valid_metric",
        type=str,
        default="acc",
        help="The evaluation metric used for model selection",
    )
    general.add_argument(
        "--display_iter",
        type=int,
        default=25,
        help="Log state after every <display_iter> batches",
    )
    general.add_argument(
        "--sort_by_len",
        type="bool",
        default=True,
        help="Sort batches by length for speed",
    )
    general.add_argument(
        "--only_test",
        type="bool",
        default=False,
        help="Only do testing",
    )

    general.add_argument(
        "--class_num", type=int, default=9, help="Class Num"
    )

    general.add_argument(
        "--save_style",
        type=bool,
        default=False,
        help="whether save style",
    )

    general.add_argument(
        "--save_path", type=str, help="style save path"
    )


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if not args.only_test:
        args.train_src_files = []
        args.train_repo_files = []

        num_dataset = len(args.dataset_name)
        if num_dataset > 1:
            if len(args.train_src) == 1:
                args.train_src = args.train_src * num_dataset
            if len(args.train_repo) == 1:
                args.train_repo = args.train_repo * num_dataset

        for i in range(num_dataset):
            dataset_name = args.dataset_name[i]
            data_dir = os.path.join(args.data_dir, dataset_name)
            train_src = os.path.join(data_dir, args.train_src[i])
            train_repo = os.path.join(data_dir, args.train_repo[i])
            if not os.path.isfile(train_src):
                raise IOError("No such file: %s" % train_src)
            if not os.path.isfile(train_repo):
                raise IOError("No such file: %s" % train_repo)

            args.train_src_files.append(train_src)
            args.train_repo_files.append(train_repo)

    args.dev_src_files = []
    args.dev_repo_files = []

    num_dataset = len(args.dataset_name)
    if num_dataset > 1:
        if len(args.dev_src) == 1:
            args.dev_src = args.dev_src * num_dataset
        if len(args.dev_repo) == 1:
            args.dev_repo = args.dev_repo * num_dataset

    for i in range(num_dataset):
        dataset_name = args.dataset_name[i]
        data_dir = os.path.join(args.data_dir, dataset_name)
        dev_src = os.path.join(data_dir, args.dev_src[i])
        dev_repo = os.path.join(data_dir, args.dev_repo[i])
        if not os.path.isfile(dev_src):
            raise IOError("No such file: %s" % dev_src)
        if not os.path.isfile(dev_repo):
            raise IOError("No such file: %s" % dev_repo)

        args.dev_src_files.append(dev_src)
        args.dev_repo_files.append(dev_repo)

    # Set model directory
    subprocess.call(["mkdir", "-p", args.model_dir])

    # Set model name
    if not args.model_name:
        import time
        import uuid

        args.model_name = (
            time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
        )

    # Set log + model file names
    suffix = "_test" if args.only_test else ""
    args.model_file = os.path.join(
        args.model_dir, args.model_name + ".mdl"
    )
    args.log_file = os.path.join(
        args.model_dir, args.model_name + suffix + ".txt"
    )
    args.pred_file = os.path.join(
        args.model_dir, args.model_name + suffix + ".json"
    )
    if args.pretrained:
        args.pretrained = os.path.join(
            args.model_dir, args.pretrained + ".mdl"
        )

    if args.use_src_word or args.use_tgt_word:
        # Make sure fix_embeddings and pretrained are consistent
        if args.fix_embeddings and not args.pretrained:
            logger.warning(
                "WARN: fix_embeddings set to False "
                "as embeddings are random."
            )
            args.fix_embeddings = False
    else:
        args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info("-" * 100)
    logger.info("Build word dictionary")
    src_dict = util.build_word_and_char_dict(
        args,
        examples=train_exs + dev_exs,
        fields=["summary"],
        dict_size=args.src_vocab_size,
        no_special_token=True,
    )
    logger.info("Num words in source = %d" % (len(src_dict)))

    # Initialize model
    model = CodClassifier(
        config.get_model_args(args), src_dict, class_num=args.class_num
    )

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats, ml_logger=None):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    ml_acc = AverageMeter()
    epoch_time = Timer()
    margin_loss = AverageMeter()

    current_epoch = global_stats["epoch"]
    pbar = tqdm(data_loader)

    pbar.set_description(
        "%s"
        % "Epoch = %d [acc = x.xx, ml_loss = x.xx, margin_loss = x.xx]"
        % current_epoch
    )

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex["batch_size"]
        if (
            args.optimizer in ["sgd", "adam"]
            and current_epoch <= args.warmup_epochs
        ):
            cur_lrate = global_stats["warmup_factor"] * (
                model.updates + 1
            )
            for param_group in model.optimizer.param_groups:
                param_group["lr"] = cur_lrate

        net_loss = model.update(ex)
        margin_loss.update(net_loss["margin_loss"], bsz)
        ml_loss.update(net_loss["ml_loss"], bsz)
        ml_acc.update(net_loss["acc"], bsz)
        log_info = (
            "Epoch = %d [acc = %.2f, ml_loss = %.2f, margin_loss = %.2f]"
            % (current_epoch, ml_acc.avg, ml_loss.avg, margin_loss.avg)
        )

        pbar.set_description("%s" % log_info)

    logger.info(
        "train: Epoch %d | acc = %.2f | ml_loss = %.2f | margin_loss = %.2f | "
        "Time for epoch = %.2f (s)"
        % (
            current_epoch,
            ml_acc.avg,
            ml_loss.avg,
            margin_loss.avg,
            epoch_time.time(),
        )
    )

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(
            args.model_file + ".checkpoint", current_epoch + 1
        )


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(
    args, data_loader, model, global_stats, mode="dev", ml_logger=None
):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    ml_loss = AverageMeter()
    ml_acc = AverageMeter()
    margin_loss = AverageMeter()

    save_style = args.save_style
    save_path = args.save_path
    hiddens = []
    repos = []

    with torch.no_grad():
        pbar = tqdm(data_loader)
        epoch = global_stats["epoch"]
        pbar.set_description(
            "%s"
            % "Test Epoch = %d [acc = x.xx, ml_loss = x.xx, margin_loss = x.xx]"
            % epoch
        )

        for _, ex in enumerate(pbar):
            bsz = ex["batch_size"]
            net_loss = model.predict(ex)
            ml_loss.update(net_loss["ml_loss"], bsz)
            margin_loss.update(net_loss["margin_loss"], bsz)
            ml_acc.update(net_loss["acc"], bsz)

            if save_style:
                repos += list(ex["repo_rep"].detach().cpu().numpy())
                hiddens += list(net_loss["hidden"])

            log_info = (
                "Test Epoch = %d [acc = %.2f, ml_loss = %.2f, margin_loss = %.2f]"
                % (epoch, ml_acc.avg, ml_loss.avg, margin_loss.avg)
            )

            pbar.set_description("%s" % log_info)

    logger.info(
        "test: Epoch %d | acc = %.2f | ml_loss = %.2f | margin_loss = %.2f | "
        "Time for epoch = %.2f (s)"
        % (
            epoch,
            ml_acc.avg,
            ml_loss.avg,
            margin_loss.avg,
            eval_time.time(),
        )
    )

    if save_style:
        repo_dicts = {}
        for idx in tqdm(range(len(hiddens))):
            if repos[idx] not in repo_dicts:
                repo_dicts[repos[idx]] = 0
            repo_idx = repo_dicts[repos[idx]]
            repo = repos[idx]
            hidden = hiddens[idx]
            os.makedirs(f"{save_path}/{repo}", exist_ok=True)
            np.save(f"{save_path}/{repo}/{repo_idx}.npy", hidden)
            repo_dicts[repos[idx]] += 1

    result = {}
    result["acc"] = ml_acc.avg
    result["loss"] = ml_loss.avg
    result["margin_loss"] = margin_loss.avg

    return result


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args, ml_logger):
    # --------------------------------------------------------------------------
    # DATA
    logger.info("-" * 100)
    logger.info("Load and process data files")

    train_exs = []
    if not args.only_test:
        args.dataset_weights = dict()
        for train_repo, train_src, dataset_name in zip(
            args.train_repo_files,
            args.train_src_files,
            args.dataset_name,
        ):
            train_files = dict()
            train_files["repo"] = train_repo
            train_files["src"] = train_src
            exs = util.load_summary(
                args,
                train_files,
                max_examples=args.max_examples,
                dataset_name=dataset_name,
            )
            lang_name = constants.DATA_LANG_MAP[dataset_name]
            args.dataset_weights[
                constants.LANG_ID_MAP[lang_name]
            ] = len(exs)
            train_exs.extend(exs)

        logger.info("Num train examples = %d" % len(train_exs))
        args.num_train_examples = len(train_exs)
        for lang_id in args.dataset_weights.keys():
            weight = (1.0 * args.dataset_weights[lang_id]) / len(
                train_exs
            )
            args.dataset_weights[lang_id] = round(weight, 2)
        logger.info("Dataset weights = %s" % str(args.dataset_weights))

    dev_exs = []
    for dev_repo, dev_src, dataset_name in zip(
        args.dev_repo_files, args.dev_src_files, args.dataset_name
    ):
        dev_files = dict()
        dev_files["repo"] = dev_repo
        dev_files["src"] = dev_src
        exs = util.load_summary(
            args,
            dev_files,
            max_examples=args.max_examples,
            dataset_name=dataset_name,
            test_split=True,
        )
        dev_exs.extend(exs)
    logger.info("Num dev examples = %d" % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info("-" * 100)
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = CodClassifier.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError("No such file: %s" % args.model_file)
            model = CodClassifier.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file):
            # Just resume training, no modifications.
            logger.info("Found a checkpoint...")
            checkpoint_file = args.model_file + ".checkpoint"
            model, start_epoch = CodClassifier.load_checkpoint(
                checkpoint_file, args.cuda
            )
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info("Using pretrained model...")
                model = CodClassifier.load(args.pretrained, args)
            else:
                logger.info("Training model from scratch...")
                model = init_from_scratch(args, train_exs, dev_exs)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info(
                "Trainable #parameters [encoder-decoder] {} [total] {}".format(
                    human_format(
                        model.network.count_encoder_parameters()
                    ),
                    human_format(model.network.count_parameters()),
                )
            )
            table = model.network.layer_wise_parameters()
            logger.info(
                "Breakdown of the trainable paramters\n%s" % table
            )

    # Use the GPU?
    if args.cuda:
        model.cuda()

    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info("-" * 100)
    logger.info("Make data loaders")

    if not args.only_test:
        train_dataset = data.SingleDataset(train_exs, model)
        if args.sort_by_len:
            train_sampler = data.SortedBatchSampler(
                train_dataset.lengths(), args.batch_size, shuffle=True
            )
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(
                train_dataset
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel,
        )

    dev_dataset = data.SingleDataset(dev_exs, model)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(
        dev_dataset
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info("-" * 100)
    logger.info(
        "CONFIG:\n%s" % json.dumps(vars(args), indent=4, sort_keys=True)
    )

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {
            "timer": Timer(),
            "epoch": 0,
            "best_valid": 0,
            "no_improvement": 0,
        }
        validate_official(args, dev_loader, model, stats, mode="test")

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info("-" * 100)
        logger.info("Starting training...")
        stats = {
            "timer": Timer(),
            "epoch": start_epoch,
            "best_valid": 0,
            "no_improvement": 0,
        }

        if (
            args.optimizer in ["sgd", "adam"]
            and args.warmup_epochs >= start_epoch
        ):
            logger.info(
                "Use warmup lrate for the %d epoch, from 0 up to %s."
                % (args.warmup_epochs, args.learning_rate)
            )
            num_batches = len(train_loader.dataset) // args.batch_size
            warmup_factor = (args.learning_rate + 0.0) / (
                num_batches * args.warmup_epochs
            )
            stats["warmup_factor"] = warmup_factor

        for epoch in range(start_epoch, args.num_epochs + 1):
            stats["epoch"] = epoch
            if (
                args.optimizer in ["sgd", "adam"]
                and epoch > args.warmup_epochs
            ):
                model.optimizer.param_groups[0]["lr"] = (
                    model.optimizer.param_groups[0]["lr"]
                    * args.lr_decay
                )

            train(args, train_loader, model, stats, ml_logger=ml_logger)
            # if epoch % 10 != 0:
            #     continue
            result = validate_official(
                args, dev_loader, model, stats, ml_logger=ml_logger
            )

            # Save best valid
            if result[args.valid_metric] > stats["best_valid"]:
                logger.info(
                    "Best valid: %s = %.2f (epoch %d, %d updates)"
                    % (
                        args.valid_metric,
                        result[args.valid_metric],
                        stats["epoch"],
                        model.updates,
                    )
                )
                model.save(args.model_file)
                stats["best_valid"] = result[args.valid_metric]
                stats["no_improvement"] = 0
            else:
                stats["no_improvement"] += 1
                if stats["no_improvement"] >= args.early_stop:
                    break


if __name__ == "__main__":
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        "Code to Natural Language Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p"
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, "a")
        else:
            logfile = logging.FileHandler(args.log_file, "w")
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info("COMMAND: %s" % " ".join(sys.argv))

    # Run!
    # task = Task.init(project_name=args.project_name, task_name=args.task_name)
    # task.connect(args)
    # ml_logger = Logger.current_logger()
    ml_logger = None
    main(args, ml_logger)
