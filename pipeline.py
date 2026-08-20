import math
import os
import json
import base64
import io
import statistics
import time
from typing import Optional

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field
from interfaze import Interfaze

load_dotenv()

interfaze = Interfaze(
    api_key=os.environ["INTERFAZE_API_KEY"],
)


class MenuItem(BaseModel):
    nm: Optional[str] = Field(description="Name of menu")
    unitprice: Optional[str] = Field(description="Unit price of menu")
    cnt: Optional[str] = Field(description="Quantity of menu")
    price: Optional[str] = Field(description="Total price of menu")
    # num 	identification # of menu
    # discountprice 	discounted price of menu
    # itemsubtotal 	price of each menu after discount applied
    # vatyn 	whether the price includes tax or not
    # etc 	others
    # sub_nm 	name of submenu
    # sub_num 	identification # of submenu
    # sub_unitprice 	unit price of submenu
    # sub_cnt 	quantity of submenu
    # sub_discountprice 	discounted price of submenu
    # sub_price 	total price of submenu
    # sub_etc 	others


class SubTotal(BaseModel):
    subtotal_price: Optional[str] = Field(description="Subtotal price")
    tax_price: Optional[str] = Field(description="Tax amount")
    service_price: Optional[str] = Field(description="Service charge")
    # discount_price 	discounted price in total
    # subtotal_count 	Total number of items
    # othersvc_price 	added charge other than service charge
    # tax_and_service 	tax + service
    # etc 	others


class Total(BaseModel):
    total_price: Optional[str] = Field(description="Total price")
    cashprice: Optional[str] = Field(description="Amount of price paid in cash")
    changeprice: Optional[str] = Field(description="Amount of change in cash")
    # total_etc 	others
    # creditcardprice 	amount of price paid in credit/debit card
    # emoneyprice 	amount of price paid in emoney, point
    # menutype_cnt 	total count of type of menu
    # menuqty_cnt 	total count of quantity


class Receipt(BaseModel):
    menu: list[MenuItem]
    sub_total: Optional[SubTotal]
    total: Optional[Total]
    # void_total
    # void_menu


def load_cord_dataset():
    # https://github.com/clovaai/cord
    # https://huggingface.co/datasets/naver-clova-ix/cord-v2
    ds = load_dataset("naver-clova-ix/cord-v2", split="test")
    return ds


def normalize_text(text: str):
    text = text.lower().strip()
    return text


def get_flattened_word_confidence_dict(precontext: list) -> dict:
    flattened_word_confidence_dict: dict[str, list[tuple[int, float]]] = {}
    index = 0

    for precontext_item in precontext:
        if precontext_item["name"] == "ocr":
            for section in precontext_item["result"]["sections"]:
                for line in section["lines"]:
                    for word in line["words"]:
                        key = normalize_text(word["text"])
                        flattened_word_confidence_dict.setdefault(key, [])
                        flattened_word_confidence_dict[key].append(
                            (index, word["confidence"])
                        )
                        index += 1

    return flattened_word_confidence_dict


def get_confidence_for_text_in_parsed(
    text: str | int | float,
    flattened_word_confidence_dict: dict,
) -> float | None:
    text = str(text)
    words = [normalize_text(w) for w in text.split() if normalize_text(w)]

    confidence_by_words: dict[str, float] = {}

    for w in words:
        if w not in flattened_word_confidence_dict:
            print(f"Word {w} does not occur")
            return

        # Ignore non unique for now
        if len(flattened_word_confidence_dict[w]) != 1:
            print(f"{w} occurs multiple times {flattened_word_confidence_dict[w]}")
            return

        confidence_by_words[w] = flattened_word_confidence_dict[w][0][1]

    return min(confidence_by_words.values())


def parse_ground_truth_to_pydantic(ground_truth: dict) -> Receipt:
    gt_parse = ground_truth["gt_parse"]
    receipt = Receipt(menu=[], sub_total=None, total=None)
    if "menu" in gt_parse:
        menu = gt_parse["menu"]
        if type(menu) == dict:
            menu = [menu]

        for menu_item in menu:
            receipt.menu.append(
                MenuItem(
                    nm=menu_item.get("nm", None),
                    cnt=menu_item.get("cnt", None),
                    price=menu_item.get("price", None),
                    unitprice=menu_item.get("unitprice", None),
                )
            )

    if "sub_total" in gt_parse:
        sub_total_dict = gt_parse["sub_total"] or {}
        receipt.sub_total = SubTotal(
            subtotal_price=sub_total_dict.get("subtotal_price", None),
            tax_price=sub_total_dict.get("tax_price", None),
            service_price=sub_total_dict.get("service_price", None),
        )

    if "total" in gt_parse:
        total_dict = gt_parse["total"] or {}
        receipt.total = Total(
            total_price=total_dict.get("total_price", None),
            cashprice=total_dict.get("cashprice", None),
            changeprice=total_dict.get("changeprice", None),
        )

    return receipt


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()

    image.save(buffer, format="png")
    image_bytes = buffer.getvalue()
    base64_img_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_img_str


def get_pred_receipt(
    img_uri: str,
) -> dict | None:
    no_of_retries = 3
    for i in range(no_of_retries):
        try:
            response = interfaze.chat.completions.parse(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract the details from this receipt. Currency unit is Indonesian Rupiah",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_uri,
                                },
                            },
                        ],
                    }
                ],
                response_format=Receipt,
            )

            pred_receipt = Receipt.model_validate(
                response.choices[0].message.parsed, strict=True
            )
            return {"receipt": pred_receipt, "precontext": response.precontext}
        except Exception as err:
            print(f"Error while calling interfaze at retry {i}: {err}")
            time.sleep(5)

    return


def run_ocr(ds: Dataset):
    result_dict: dict[str, tuple[Receipt, Receipt]] = {}

    for x in ds:
        img_uri = f"data:image/png;base64,{image_to_base64(x['image'])}"

        ground_truth = json.loads(x["ground_truth"])
        image_id = ground_truth["meta"]["image_id"]
        print(f"image_id: {image_id}")
        ground_truth_receipt = parse_ground_truth_to_pydantic(ground_truth)

        pred_res = get_pred_receipt(img_uri)

        if pred_res is None:
            continue

        result_dict[image_id] = (ground_truth_receipt, pred_res)

    return result_dict


def emit_field_results_from_receipt(
    img_id: str,
    ground_truth_receipt: Receipt,
    pred_res: dict,
) -> list[tuple]:
    field_results = []
    # img_id, category_key, orig_alue, pred_value, confidence
    ground_truth_menu_map = {x.nm: x for x in ground_truth_receipt.menu if x.nm}
    pred_receipt: Receipt = pred_res["receipt"]
    precontext: dict | None = pred_res["precontext"]

    if precontext is None:
        return []

    flattened_word_confidence_dict = get_flattened_word_confidence_dict(precontext)

    for menu_item in pred_receipt.menu:
        nm = normalize_text(menu_item.nm)

        if nm not in ground_truth_menu_map:
            continue

        ground_truth_menu_item = ground_truth_menu_map[nm]

        for k, v in menu_item:
            if v is None:
                continue

            pred_v = normalize_text(v)
            actual_v = getattr(ground_truth_menu_item, k, None)

            confidence = get_confidence_for_text_in_parsed(
                pred_v, flattened_word_confidence_dict
            )

            if confidence is None:
                continue

            field_results.append(
                (
                    img_id,
                    f"menu_item.{k}",
                    actual_v,
                    pred_v,
                    confidence,
                )
            )

    for receipt_k, receipt_v in pred_receipt:
        if receipt_k == "menu":
            continue

        if receipt_v is None:
            continue

        ground_truth_v = getattr(ground_truth_receipt, receipt_k, None)

        for k, v in receipt_v:
            if v is None:
                continue

            pred_v = normalize_text(v)
            actual_v = getattr(ground_truth_v, k, None) if ground_truth_v else None

            confidence = get_confidence_for_text_in_parsed(
                pred_v, flattened_word_confidence_dict
            )

            if confidence is None:
                continue

            field_results.append(
                (
                    img_id,
                    f"{receipt_k}.{k}",
                    actual_v,
                    pred_v,
                    confidence,
                )
            )

    return field_results


def ece_pipeline():
    ds = load_cord_dataset()
    sub_dataset = ds.select(range(3))
    result_dict = run_ocr(sub_dataset)
    field_results = []

    for k, v in result_dict.items():
        field_results.extend(emit_field_results_from_receipt(k, v[0], v[1]))

    return field_results


def calculate_ece(field_results: list[tuple]):
    bins = [round(x * 0.1, 1) for x in range(10)]
    field_results_by_bins = [[] for _ in bins]

    for fr in field_results:
        confidence = round(fr[4], 2)
        index = math.floor(confidence * 10)

        field_results_by_bins[index].append(fr)

    ece = 0
    stats = {
        "accuracies": [],
        "avg_confidences": [],
    }

    for i in range(10):
        if len(field_results_by_bins[i]) == 0:
            stats["accuracies"].append(0)
            stats["avg_confidences"].append(0)
            continue

        average_confidence = statistics.mean([x[4] for x in field_results_by_bins[i]])
        accuracy = statistics.mean(
            [int(x[2] == x[3]) for x in field_results_by_bins[i]]
        )
        ece += (len(field_results_by_bins[i]) / len(field_results)) * abs(
            average_confidence - accuracy
        )

        stats["accuracies"].append(accuracy)
        stats["avg_confidences"].append(average_confidence)

    return {
        "ece": ece,
        "field_results_by_bins": field_results_by_bins,
        "bins": bins,
        "stats": stats,
    }


def draw_reliablity_diagram(ece_data: dict):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(
        ece_data["bins"],
        ece_data["stats"]["accuracies"],
        color="lightblue",
        width=0.1,
        align="edge",
        edgecolor="black",
    )
    ax.plot([0, 1], [0, 1], color="red", linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability_diagram")

    plt.show()
    fig.savefig("reliability_diagram.png")
