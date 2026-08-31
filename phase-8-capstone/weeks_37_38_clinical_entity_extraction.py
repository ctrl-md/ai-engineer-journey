import scispacy
import spacy

from texts import (
    text1,
    text2,
    text3,
    text4,
    text5,
    ground_truth_text1,
    ground_truth_text2,
    ground_truth_text3,
    ground_truth_text4,
    ground_truth_text5,
)


def load_nlp_model(model_name):
    nlp = spacy.load(model_name)
    return nlp


def strip_boilerplate(text):
    text_list = text.splitlines()
    new_list = [
        line.strip()
        for line in text_list
        if not (
            line.startswith("Sample Name:")
            or line.startswith("Medical Specialty:")
            or line.startswith("Description:")
            or line.startswith("(Medical Transcription Sample Report)")
            or line.startswith("Intended for:")
        )
    ]

    text = "\n".join(new_list).strip()
    return text


def is_negated(entity, trigger_words, window=6):
    span = entity.sent
    for token in span:
        if token.i < entity.start and token.i >= entity.start - window:
            if token.text.lower() in trigger_words:
                return True

    if entity[0].text.lower() in trigger_words:
        return True
    return False


def extract_entities_with_context(text, model_name, trigger_words, window=6):
    text = strip_boilerplate(text)
    nlp = load_nlp_model(model_name)
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        negated = is_negated(ent, trigger_words, window)
        if not negated:
            entities.append({"text": ent.text, "label": ent.label_})
    return entities


def compute_precision(predicted_entities, ground_truth_entities):
    ground_truth_entities = list(ground_truth_entities)
    text_matching_entities = 0
    label_matching_entities = 0
    num_ground_truth_entities = len(ground_truth_entities)
    for predicted_entity in predicted_entities:
        for ground_truth_entity in ground_truth_entities:
            if (
                predicted_entity["text"]
                .lower()
                .find(ground_truth_entity["text"].lower())
                != -1
            ):
                text_matching_entities += 1
                if (
                    predicted_entity["label"]
                    .lower()
                    .find(ground_truth_entity["label"].lower())
                    != -1
                ):
                    label_matching_entities += 1
                ground_truth_entities.remove(ground_truth_entity)
                break
    extracted_typed = (
        label_matching_entities / text_matching_entities
        if text_matching_entities
        else 0
    )
    text_precision = (
        text_matching_entities / len(predicted_entities) if predicted_entities else 0
    )
    label_precision = (
        label_matching_entities / len(predicted_entities) if predicted_entities else 0
    )
    text_recall = (
        text_matching_entities / num_ground_truth_entities
        if num_ground_truth_entities
        else 0
    )
    label_recall = (
        label_matching_entities / num_ground_truth_entities
        if num_ground_truth_entities
        else 0
    )
    return extracted_typed, text_precision, label_precision, text_recall, label_recall


def to_untyped(ground_truth):
    return [{"text": e["text"], "label": "ENTITY"} for e in ground_truth]


if __name__ == "__main__":
    trigger_words = ["no", "not", "without", "denies", "denied", "negative"]
    textlist = [text1, text2, text3, text4, text5]
    typed_ground_truth_list = [
        ground_truth_text1,
        ground_truth_text2,
        ground_truth_text3,
        ground_truth_text4,
        ground_truth_text5,
    ]
    untyped_ground_truth_list = [
        to_untyped(ground_truth_text1),
        to_untyped(ground_truth_text2),
        to_untyped(ground_truth_text3),
        to_untyped(ground_truth_text4),
        to_untyped(ground_truth_text5),
    ]
    for index, text in enumerate(textlist):
        untyped_extract_entities = extract_entities_with_context(
            text, "en_core_sci_sm", trigger_words, window=10
        )
        typed_extract_entities = extract_entities_with_context(
            text, "en_ner_bc5cdr_md", trigger_words, window=10
        )
        (
            untyped_extracted_typed,
            untyped_text_precision,
            _,
            untyped_text_recall,
            _,
        ) = compute_precision(
            untyped_extract_entities, untyped_ground_truth_list[index]
        )
        (
            typed_extracted_typed,
            typed_text_precision,
            typed_label_precision,
            typed_text_recall,
            typed_label_recall,
        ) = compute_precision(typed_extract_entities, typed_ground_truth_list[index])

        print(f"Text {index + 1}:")
        print(
            f"  Untyped - Extracted Typed: {untyped_extracted_typed}, Text Precision: {untyped_text_precision}, Text Recall: {untyped_text_recall}"
        )
        print(
            f"  Typed - Extracted Typed: {typed_extracted_typed}, Text Precision: {typed_text_precision}, Label Precision: {typed_label_precision}, Text Recall: {typed_text_recall}, Label Recall: {typed_label_recall}"
        )
