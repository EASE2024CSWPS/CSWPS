import torch
import torch.nn as nn
from c2nl.encoders.transformer import TransformerEncoder
from c2nl.inputters import constants
from c2nl.modules.char_embedding import CharEmbedding
from c2nl.modules.embeddings import Embeddings
from c2nl.modules.highway import Highway
from c2nl.utils.misc import sequence_mask
from prettytable import PrettyTable


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()

        self.enc_input_size = 0

        # at least one of word or char embedding options should be True
        assert args.use_src_word or args.use_src_char

        self.use_src_word = args.use_src_word
        if self.use_src_word:
            self.src_word_embeddings = Embeddings(
                args.emsize, args.src_vocab_size, constants.PAD
            )
            self.enc_input_size += args.emsize

        self.use_src_char = args.use_src_char
        if self.use_src_char:
            assert len(args.filter_size) == len(args.nfilters)
            self.src_char_embeddings = CharEmbedding(
                args.n_characters,
                args.char_emsize,
                args.filter_size,
                args.nfilters,
            )
            self.enc_input_size += sum(list(map(int, args.nfilters)))
            self.src_highway_net = Highway(
                self.enc_input_size, num_layers=2
            )

        self.src_pos_emb = args.src_pos_emb
        self.no_relative_pos = all(
            v == 0 for v in args.max_relative_pos
        )

        if self.src_pos_emb and self.no_relative_pos:
            self.src_pos_embeddings = nn.Embedding(
                args.max_src_len, self.enc_input_size
            )
        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(
        self,
        sequence,
        sequence_char,
        sequence_type=None,
        mode="encoder",
        step=None,
    ):
        word_rep = None
        if self.use_src_word:
            word_rep = self.src_word_embeddings(
                sequence.unsqueeze(2)
            )  # B x P x d
        if self.use_src_char:
            char_rep = self.src_char_embeddings(
                sequence_char
            )  # B x P x f
            if word_rep is None:
                word_rep = char_rep
            else:
                word_rep = torch.cat(
                    (word_rep, char_rep), 2
                )  # B x P x d+f
            word_rep = self.src_highway_net(word_rep)  # B x P x d+f

        if self.src_pos_emb and self.no_relative_pos:
            pos_enc = torch.arange(start=0, end=word_rep.size(1)).type(
                torch.LongTensor
            )
            pos_enc = pos_enc.expand(*word_rep.size()[:-1])
            if word_rep.is_cuda:
                pos_enc = pos_enc.cuda()
            pos_rep = self.src_pos_embeddings(pos_enc)
            word_rep = word_rep + pos_rep

        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    def __init__(self, args, input_size):
        super(Encoder, self).__init__()

        self.transformer = TransformerEncoder(
            num_layers=args.nlayers,
            d_model=input_size,
            heads=args.num_head,
            d_k=args.d_k,
            d_v=args.d_v,
            d_ff=args.d_ff,
            dropout=args.trans_drop,
            max_relative_positions=args.max_relative_pos,
            use_neg_dist=args.use_neg_dist,
        )
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self, input, input_len):
        layer_outputs, _ = self.transformer(
            input, input_len
        )  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(
                layer_outputs, dim=2
            )  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = torch.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(
                output.transpose(2, 3), layer_scores.unsqueeze(3)
            ).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Transformer(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, args):
        """ "Constructor of the class."""
        super(Transformer, self).__init__()

        self.name = "TransformerClassifier"
        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers

        self.class_num = args.class_num
        self.embedder = Embedder(args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)

        self.v0 = nn.Parameter(
            torch.randn(self.embedder.enc_input_size, 1)
        )  # [Dim, 1]

        self.fc_layer = nn.Linear(
            self.embedder.enc_input_size, self.class_num
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.margin_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0
            - torch.cosine_similarity(x, y)
        )

    def forward(
        self,
        src_word_rep,
        src_char_rep,
        src_len,
        repo_ids,
        pos_src_word_rep,
        pos_src_char_rep,
        pos_src_len,
        pos_repo_ids,
        neg_src_word_rep,
        neg_src_char_rep,
        neg_src_len,
        neg_repo_ids,
        return_hidden=False,
        test=False,
    ):
        # embed and encode the source sequence
        src_rep = self.embedder(
            src_word_rep, src_char_rep, mode="encoder"
        )

        memory_bank, _ = self.encoder(
            src_rep, src_len
        )  # B x seq_len x h
        # [B,seq_len, h] * [h,1]
        attention_score = (
            memory_bank @ self.v0
        ).squeeze()  # [B,seq_len)
        src_pad_mask = ~sequence_mask(src_len, max_len=src_rep.size(1))
        mask = src_pad_mask.byte()  # Make it broadcastable.
        attention_score.data.masked_fill_(mask, -1e8)

        attention_score = torch.softmax(attention_score, dim=-1)

        hidden_state = torch.sum(
            memory_bank * attention_score.unsqueeze(-1), dim=1
        )
        hidden_state = hidden_state.squeeze(-1)  # [B*dim]

        if not test:
            pos_src_rep = self.embedder(
                pos_src_word_rep, pos_src_char_rep, mode="encoder"
            )
            neg_src_rep = self.embedder(
                neg_src_word_rep, neg_src_char_rep, mode="encoder"
            )
            pos_memory_bank, _ = self.encoder(
                pos_src_rep, pos_src_len
            )  # B x seq_len x h
            neg_memory_bank, _ = self.encoder(
                neg_src_rep, neg_src_len
            )  # B x seq_len x h
            pos_attention_score = (
                pos_memory_bank @ self.v0
            ).squeeze()  # [B,seq_len)
            neg_attention_score = (
                neg_memory_bank @ self.v0
            ).squeeze()  # [B,seq_len)

            pos_src_pad_mask = ~sequence_mask(
                pos_src_len, max_len=pos_src_rep.size(1)
            )
            pos_mask = pos_src_pad_mask.byte()  # Make it broadcastable.
            pos_attention_score.data.masked_fill_(pos_mask, -1e8)

            neg_src_pad_mask = ~sequence_mask(
                neg_src_len, max_len=neg_src_rep.size(1)
            )
            neg_mask = neg_src_pad_mask.byte()  # Make it broadcastable.
            neg_attention_score.data.masked_fill_(neg_mask, -1e8)
            pos_attention_score = torch.softmax(
                pos_attention_score, dim=-1
            )
            neg_attention_score = torch.softmax(
                neg_attention_score, dim=-1
            )
            pos_hidden_state = torch.sum(
                pos_memory_bank * pos_attention_score.unsqueeze(-1),
                dim=1,
            )
            pos_hidden_state = pos_hidden_state.squeeze(-1)  # [B*dim]

            neg_hidden_state = torch.sum(
                neg_memory_bank * neg_attention_score.unsqueeze(-1),
                dim=1,
            )
            neg_hidden_state = neg_hidden_state.squeeze(-1)  # [B*dim]

            total_hidden = torch.cat(
                [hidden_state, pos_hidden_state, neg_hidden_state],
                dim=0,
            )  # [3B*dim]
            total_repo_ids = torch.cat(
                [repo_ids, pos_repo_ids, neg_repo_ids], dim=0
            )  # [3B]

            margin_loss = self.margin_loss(
                hidden_state, pos_hidden_state, neg_hidden_state
            )
        else:
            total_hidden = hidden_state
            total_repo_ids = repo_ids
            margin_loss = 0.0

        output = self.fc_layer(total_hidden)  # [B, class_num]
        ml_loss = self.criterion(output, total_repo_ids)

        # print(margin_loss)
        # calculate acc
        logits = torch.softmax(output, dim=-1)
        _, predicted_labels = torch.max(logits, dim=1)
        # print(hidden_state, pos_hidden_state, neg_hidden_state)
        # print([repo_ids, pos_repo_ids, neg_repo_ids])
        # print(predicted_labels, total_repo_ids)
        # quit(-1)
        correct = (predicted_labels == total_repo_ids).sum().item()
        accuracy = correct / total_hidden.size(0)

        res = {}
        res["ml_loss"] = ml_loss.mean()
        res["margin_loss"] = margin_loss
        res["acc"] = accuracy * 100

        if return_hidden:
            res["hidden_state"] = hidden_state

        return res

    def count_parameters(self):
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row(
                    [
                        name,
                        str(list(parameters.shape)),
                        parameters.numel(),
                    ]
                )
        return table
