def chrf_score(pred: str, ref: str) -> float:
    import sacrebleu

    return sacrebleu.sentence_chrf(pred, [ref]).score


def meteor_score(pred: str, ref: str) -> float:
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score as _nltk_meteor

    return _nltk_meteor([word_tokenize(ref)], word_tokenize(pred))


def bertscore_batch(preds: list[str], refs: list[str], lang: str = "en") -> dict:
    """Lazy import: pulls in torch + downloads a ~1.4GB scoring model on first use."""
    from bert_score import score

    precision, recall, f1 = score(preds, refs, lang=lang)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }
