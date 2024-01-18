import copy
import logging

import torch
import torch.optim as optim
from c2nl.config import override_model_args
from c2nl.models.transformer import Transformer
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


class CodClassifier(object):
    """High level model that handles initializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, class_num=None, state_dict=None):
        # Book-keeping.
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.class_num = class_num
        self.args.class_num = class_num

        self.network = Transformer(self.args)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if "fixed_embedding" in state_dict:
                fixed_embedding = state_dict.pop("fixed_embedding")
                self.network.load_state_dict(state_dict)
                self.network.register_buffer(
                    "fixed_embedding", fixed_embedding
                )
            else:
                self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            self.network.embedder.src_word_embeddings.fix_word_lut()

        if self.args.optimizer == "sgd":
            parameters = [
                p for p in self.network.parameters() if p.requires_grad
            ]
            self.optimizer = optim.SGD(
                parameters,
                self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            parameters = [
                p for p in self.network.parameters() if p.requires_grad
            ]
            self.optimizer = optim.Adam(
                parameters,
                self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        else:
            raise RuntimeError(
                "Unsupported optimizer: %s" % self.args.optimizer
            )

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError("No optimizer set.")

        # Train mode
        self.network.train()

        src_word_rep = ex["src_word_rep"]
        src_char_rep = ex["src_char_rep"]
        src_len = ex["src_len"]
        repo_rep = ex["repo_rep"]

        pos_src_word_rep = ex["pos_src_word_rep"]
        pos_src_char_rep = ex["pos_src_char_rep"]
        pos_src_len = ex["pos_src_len"]
        pos_repo_rep = ex["pos_repo_rep"]

        neg_src_word_rep = ex["neg_src_word_rep"]
        neg_src_char_rep = ex["neg_src_char_rep"]
        neg_src_len = ex["neg_src_len"]
        neg_repo_rep = ex["neg_repo_rep"]

        if any(l is None for l in ex["language"]):
            ex_weights = None
        else:
            ex_weights = [
                self.args.dataset_weights[lang]
                for lang in ex["language"]
            ]
            ex_weights = torch.FloatTensor(ex_weights)

        if self.use_cuda:
            pos_src_len = pos_src_len.cuda(non_blocking=True)
            neg_src_len = neg_src_len.cuda(non_blocking=True)
            src_len = src_len.cuda(non_blocking=True)
            if src_word_rep is not None:
                src_word_rep = src_word_rep.cuda(non_blocking=True)
                pos_src_word_rep = pos_src_word_rep.cuda(
                    non_blocking=True
                )
                neg_src_word_rep = neg_src_word_rep.cuda(
                    non_blocking=True
                )
            if src_char_rep is not None:
                src_char_rep = src_char_rep.cuda(non_blocking=True)
                pos_src_char_rep = pos_src_char_rep.cuda(
                    non_blocking=True
                )
                neg_src_char_rep = neg_src_char_rep.cuda(
                    non_blocking=True
                )
            if repo_rep is not None:
                repo_rep = repo_rep.cuda(non_blocking=True)
                pos_repo_rep = pos_repo_rep.cuda(non_blocking=True)
                neg_repo_rep = neg_repo_rep.cuda(non_blocking=True)

        # Run forward
        res = self.network(
            src_word_rep=src_word_rep,
            src_char_rep=src_char_rep,
            src_len=src_len,
            repo_ids=repo_rep,
            pos_src_word_rep=pos_src_word_rep,
            pos_src_char_rep=pos_src_char_rep,
            pos_src_len=pos_src_len,
            pos_repo_ids=pos_repo_rep,
            neg_src_word_rep=neg_src_word_rep,
            neg_src_char_rep=neg_src_char_rep,
            neg_src_len=neg_src_len,
            neg_repo_ids=neg_repo_rep,
            return_hidden=False,
        )

        loss = (
            res["ml_loss"].mean() if self.parallel else res["ml_loss"]
        )
        margin_loss = (
            res["margin_loss"].mean()
            if self.parallel
            else res["margin_loss"]
        )
        acc = res["acc"].mean() if self.parallel else res["acc"]
        ml_loss = loss.item()
        ml_margin_loss = margin_loss.item()

        loss = loss + 2 * margin_loss

        loss.backward()

        clip_grad_norm_(
            self.network.parameters(), self.args.grad_clipping
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            "ml_loss": ml_loss,
            "margin_loss": ml_margin_loss,
            "acc": acc,
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
            src_raw: raw source (passage); required to replace `unk` term
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        src_word_rep = ex["src_word_rep"]
        src_char_rep = ex["src_char_rep"]
        src_len = ex["src_len"]
        repo_rep = ex["repo_rep"]

        pos_src_word_rep = ex["pos_src_word_rep"]
        pos_src_char_rep = ex["pos_src_char_rep"]
        pos_src_len = ex["pos_src_len"]
        pos_repo_rep = ex["pos_repo_rep"]

        neg_src_word_rep = ex["neg_src_word_rep"]
        neg_src_char_rep = ex["neg_src_char_rep"]
        neg_src_len = ex["neg_src_len"]
        neg_repo_rep = ex["neg_repo_rep"]

        if any(l is None for l in ex["language"]):
            ex_weights = None
        else:
            ex_weights = [
                self.args.dataset_weights[lang]
                for lang in ex["language"]
            ]
            ex_weights = torch.FloatTensor(ex_weights)

        if self.use_cuda:
            pos_src_len = pos_src_len.cuda(non_blocking=True)
            neg_src_len = neg_src_len.cuda(non_blocking=True)
            src_len = src_len.cuda(non_blocking=True)
            if src_word_rep is not None:
                src_word_rep = src_word_rep.cuda(non_blocking=True)
                pos_src_word_rep = pos_src_word_rep.cuda(
                    non_blocking=True
                )
                neg_src_word_rep = neg_src_word_rep.cuda(
                    non_blocking=True
                )
            if src_char_rep is not None:
                src_char_rep = src_char_rep.cuda(non_blocking=True)
                pos_src_char_rep = pos_src_char_rep.cuda(
                    non_blocking=True
                )
                neg_src_char_rep = neg_src_char_rep.cuda(
                    non_blocking=True
                )
            if repo_rep is not None:
                repo_rep = repo_rep.cuda(non_blocking=True)
                pos_repo_rep = pos_repo_rep.cuda(non_blocking=True)
                neg_repo_rep = neg_repo_rep.cuda(non_blocking=True)

        # Run forward
        res = self.network(
            src_word_rep=src_word_rep,
            src_char_rep=src_char_rep,
            src_len=src_len,
            repo_ids=repo_rep,
            pos_src_word_rep=pos_src_word_rep,
            pos_src_char_rep=pos_src_char_rep,
            pos_src_len=pos_src_len,
            pos_repo_ids=pos_repo_rep,
            neg_src_word_rep=neg_src_word_rep,
            neg_src_char_rep=neg_src_char_rep,
            neg_src_len=neg_src_len,
            neg_repo_ids=neg_repo_rep,
            return_hidden=True,
            test=True,
        )

        loss = (
            res["ml_loss"].mean() if self.parallel else res["ml_loss"]
        )
        margin_loss = (
            res["margin_loss"].mean()
            if self.parallel
            else res["margin_loss"]
        )
        acc = res["acc"].mean() if self.parallel else res["acc"]
        ml_loss = loss.item()
        ml_margin_loss = margin_loss

        return {
            "ml_loss": ml_loss,
            "margin_loss": ml_margin_loss,
            "acc": acc,
            "hidden": res["hidden_state"].detach().cpu().numpy(),
        }

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if "fixed_embedding" in state_dict:
            state_dict.pop("fixed_embedding")
        params = {
            "state_dict": state_dict,
            "src_dict": self.src_dict,
            "class_num": self.class_num,
            "args": self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning("WARN: Saving failed... continuing anyway.")

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            "state_dict": network.state_dict(),
            "src_dict": self.src_dict,
            "args": self.args,
            "epoch": epoch,
            "updates": self.updates,
            "class_num": self.class_num,
            "optimizer": self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning("WARN: Saving failed... continuing anyway.")

    @staticmethod
    def load(filename, new_args=None):
        logger.info("Loading model %s" % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params["src_dict"]
        state_dict = saved_params["state_dict"]
        args = saved_params["args"]
        class_num = saved_params["class_num"]
        if new_args:
            args = override_model_args(args, new_args)
        return CodClassifier(args, src_dict, class_num, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info("Loading model %s" % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params["src_dict"]
        state_dict = saved_params["state_dict"]
        epoch = saved_params["epoch"]
        updates = saved_params["updates"]
        optimizer = saved_params["optimizer"]
        args = saved_params["args"]
        class_num = saved_params["class_num"]
        model = CodClassifier(args, src_dict, class_num, state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
