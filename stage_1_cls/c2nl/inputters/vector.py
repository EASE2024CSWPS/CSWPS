import numpy as np
import torch


def vectorize(ex, pos_ex, neg_ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict

    summary = ex["summary"]
    vectorized_ex = dict()
    vectorized_ex["id"] = summary.id
    vectorized_ex["language"] = ex["lang"]

    vectorized_ex["src"] = summary.text
    vectorized_ex["src_tokens"] = summary.tokens
    vectorized_ex["src_word_rep"] = torch.LongTensor(
        summary.vectorize(word_dict=src_dict)
    )
    vectorized_ex["src_char_rep"] = None
    vectorized_ex["use_src_word"] = model.args.use_src_word
    vectorized_ex["use_src_char"] = model.args.use_src_char
    vectorized_ex["repo"] = int(ex["repo"])

    pos_summary = pos_ex["summary"]

    vectorized_ex["pos_src"] = pos_summary.text
    vectorized_ex["pos_src_tokens"] = pos_summary.tokens
    vectorized_ex["pos_src_word_rep"] = torch.LongTensor(
        pos_summary.vectorize(word_dict=src_dict)
    )
    vectorized_ex["pos_src_char_rep"] = None
    vectorized_ex["pos_repo"] = int(pos_ex["repo"])

    neg_summary = neg_ex["summary"]

    vectorized_ex["neg_src"] = neg_summary.text
    vectorized_ex["neg_src_tokens"] = neg_summary.tokens
    vectorized_ex["neg_src_word_rep"] = torch.LongTensor(
        neg_summary.vectorize(word_dict=src_dict)
    )
    vectorized_ex["neg_src_char_rep"] = None
    vectorized_ex["neg_repo"] = int(neg_ex["repo"])

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]["use_src_word"]
    use_src_char = batch[0]["use_src_char"]
    ids = [ex["id"] for ex in batch]
    language = [ex["language"] for ex in batch]

    src_words = [ex["src_word_rep"] for ex in batch]
    src_chars = [ex["src_char_rep"] for ex in batch]

    pos_src_words = [ex["pos_src_word_rep"] for ex in batch]
    neg_src_words = [ex["neg_src_word_rep"] for ex in batch]

    max_src_len = max([q.size(0) for q in src_words])
    pos_max_src_len = max([q.size(0) for q in pos_src_words])
    neg_max_src_len = max([q.size(0) for q in neg_src_words])

    if use_src_char:
        max_char_in_src_token = src_chars[0].size(1)

    src_len_rep = torch.zeros(batch_size, dtype=torch.long)
    src_word_rep = (
        torch.zeros(batch_size, max_src_len, dtype=torch.long)
        if use_src_word
        else None
    )

    pos_src_len_rep = torch.zeros(batch_size, dtype=torch.long)
    pos_src_word_rep = (
        torch.zeros(batch_size, pos_max_src_len, dtype=torch.long)
        if use_src_word
        else None
    )

    neg_src_len_rep = torch.zeros(batch_size, dtype=torch.long)
    neg_src_word_rep = (
        torch.zeros(batch_size, neg_max_src_len, dtype=torch.long)
        if use_src_word
        else None
    )

    src_char_rep = (
        torch.zeros(
            batch_size,
            max_src_len,
            max_char_in_src_token,
            dtype=torch.long,
        )
        if use_src_char
        else None
    )

    for i in range(batch_size):
        src_len_rep[i] = src_words[i].size(0)
        pos_src_len_rep[i] = pos_src_words[i].size(0)
        neg_src_len_rep[i] = neg_src_words[i].size(0)
        if use_src_word:
            src_word_rep[i, : src_words[i].size(0)].copy_(src_words[i])
            pos_src_word_rep[i, : pos_src_words[i].size(0)].copy_(
                pos_src_words[i]
            )
            neg_src_word_rep[i, : neg_src_words[i].size(0)].copy_(
                neg_src_words[i]
            )
        if use_src_char:
            src_char_rep[i, : src_chars[i].size(0), :].copy_(
                src_chars[i]
            )

    return {
        "ids": ids,
        "language": language,
        "batch_size": batch_size,
        "src_word_rep": src_word_rep,
        "src_char_rep": src_char_rep,
        "pos_src_word_rep": pos_src_word_rep,
        "pos_src_char_rep": None,
        "neg_src_word_rep": neg_src_word_rep,
        "neg_src_char_rep": None,
        "src_len": src_len_rep,
        "src_text": [ex["src"] for ex in batch],
        "src_tokens": [ex["src_tokens"] for ex in batch],
        "repo_rep": torch.LongTensor([ex["repo"] for ex in batch]),
        "pos_src_len": pos_src_len_rep,
        "pos_src_text": [ex["pos_src"] for ex in batch],
        "pos_src_tokens": [ex["pos_src_tokens"] for ex in batch],
        "pos_repo_rep": torch.LongTensor(
            [ex["pos_repo"] for ex in batch]
        ),
        "neg_src_len": neg_src_len_rep,
        "neg_src_text": [ex["neg_src"] for ex in batch],
        "neg_src_tokens": [ex["neg_src_tokens"] for ex in batch],
        "neg_repo_rep": torch.LongTensor(
            [ex["neg_repo"] for ex in batch]
        ),
    }
