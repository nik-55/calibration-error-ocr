import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import base64
import io
from pathlib import Path
import re
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
    cnt: Optional[str] = Field(description="Quantity of menu")
    price: Optional[str] = Field(description="Total price of menu")
    # unitprice: Optional[str] = Field(description="Unit price of menu")
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
    creditcardprice: Optional[str] = Field(
        description="Amount of price paid in credit/debit card"
    )
    # total_etc 	others
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


def loose(s: str):
    return re.sub(r"[.,\s]", "", s) if isinstance(s, str) else s


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
    losse_flattened_word_confidence_dict = {
        loose(k): v for k, v in flattened_word_confidence_dict.items()
    }

    for w in words:
        w = loose(w)
        if (
            w not in losse_flattened_word_confidence_dict
            or len(losse_flattened_word_confidence_dict[w]) == 0
        ):
            print(f"Word {w} does not occur")
            return

        # Non unique for now
        if len(losse_flattened_word_confidence_dict[w]) > 1:
            print(
                f"{w} occurs multiple times {losse_flattened_word_confidence_dict[w]}"
            )

        confidence_by_words[w] = min(
            [x[1] for x in losse_flattened_word_confidence_dict[w]]
        )

    return min(confidence_by_words.values())


def parse_ground_truth_to_pydantic(ground_truth: dict) -> Receipt:
    gt_parse = ground_truth["gt_parse"]
    receipt = Receipt(menu=[], sub_total=None, total=None)
    if "menu" in gt_parse:
        menu = gt_parse["menu"]
        if type(menu) == dict:
            menu = [menu]

        for menu_item in menu:
            for k, v in menu_item.items():
                if v is not None and isinstance(v, str):
                    menu_item[k] = normalize_text(v)

            receipt.menu.append(
                MenuItem(
                    nm=menu_item.get("nm", None),
                    cnt=menu_item.get("cnt", None),
                    price=menu_item.get("price", None),
                )
            )

    if "sub_total" in gt_parse:
        sub_total_dict = gt_parse["sub_total"] or {}
        for k, v in sub_total_dict.items():
            if v is not None:
                if isinstance(v, list):
                    v = v[0] if len(v) > 0 else []

                if isinstance(v, str):
                    sub_total_dict[k] = normalize_text(v)

        receipt.sub_total = SubTotal(
            subtotal_price=sub_total_dict.get("subtotal_price", None),
            tax_price=sub_total_dict.get("tax_price", None),
            service_price=sub_total_dict.get("service_price", None),
        )

    if "total" in gt_parse:
        total_dict = gt_parse["total"] or {}

        for k, v in total_dict.items():
            if v is not None:
                if isinstance(v, list):
                    v = v[0] if len(v) > 0 else []

                if isinstance(v, str):
                    total_dict[k] = normalize_text(v)

        receipt.total = Total(
            total_price=total_dict.get("total_price", None),
            cashprice=total_dict.get("cashprice", None),
            changeprice=total_dict.get("changeprice", None),
            creditcardprice=total_dict.get("creditcardprice", None),
        )

    return receipt


def img_transform_and_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()

    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

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
                                "text": """
Extract the details from this receipt. Copy every value exactly as printed —
keep the original digit grouping and separators (write 11.000 or 11,000 exactly
as it appears), and do not reformat, round, convert, or compute any value.
If a field is not printed on the receipt, return null.
Currency is Indonesian Rupiah.
""",
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


def do_ocr(i: int, ds: Dataset, ocr_res_base_dir: Path, ignore_cache: bool):
    image_id = None
    x = ds[i]

    try:
        img_uri = f"data:image/png;base64,{img_transform_and_to_base64(x['image'])}"

        ground_truth = json.loads(x["ground_truth"])
        image_id = ground_truth["meta"]["image_id"]

        ocr_res_path = ocr_res_base_dir / f"{image_id}.json"
        print(f"image_id: {image_id} ocr_res_path: {ocr_res_path}")

        if ocr_res_path.exists() and ignore_cache is False:
            ocr_data = json.loads(ocr_res_path.read_text())
            return {
                image_id: (
                    Receipt.model_validate(ocr_data["ground_truth_receipt"]),
                    {
                        "receipt": Receipt.model_validate(
                            ocr_data["pred_res"]["receipt"]
                        ),
                        "precontext": ocr_data["pred_res"]["precontext"],
                    },
                )
            }

        ground_truth_receipt = parse_ground_truth_to_pydantic(ground_truth)

        if ground_truth_receipt is None:
            print(f"ground_truth_receipt is None: {image_id}")
            return

        pred_res = get_pred_receipt(img_uri)

        if pred_res is None:
            print(f"pred_res is None: {image_id}")
            return

        ocr_res_path.write_text(
            json.dumps(
                {
                    "image_id": image_id,
                    "ground_truth_receipt": ground_truth_receipt.model_dump(),
                    "pred_res": {
                        "receipt": pred_res["receipt"].model_dump(),
                        "precontext": pred_res["precontext"],
                    },
                },
                indent=2,
            )
        )

        return {image_id: (ground_truth_receipt, pred_res)}
    except Exception as err:
        print(f"Error ocr for {x} with {image_id}: {err}")


def run_ocr(ds: Dataset, ignore_cache: bool = False):
    result_dict: dict[str, tuple[Receipt, dict]] = {}
    ocr_res_base_dir = Path("assets/interfaze_ocr_results")
    ocr_res_base_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(do_ocr, i, ds, ocr_res_base_dir, ignore_cache)
            for i in range(len(ds))
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                result_dict.update(result)

    return result_dict


def emit_field_results_from_receipt(
    img_id: str,
    ground_truth_receipt: Receipt,
    pred_res: dict,
) -> list[tuple]:
    field_results = []
    # img_id, category_key, orig_alue, pred_value, confidence
    ground_truth_menu_map = {
        loose(x.nm): x for x in ground_truth_receipt.menu if loose(x.nm)
    }
    pred_receipt: Receipt = pred_res["receipt"]
    precontext: dict | None = pred_res["precontext"]

    if precontext is None:
        return []

    flattened_word_confidence_dict = get_flattened_word_confidence_dict(precontext)

    for menu_item in pred_receipt.menu:
        if menu_item.nm is None:
            continue

        nm = loose(normalize_text(menu_item.nm))

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


def draw_reliablity_diagram(metric_data: dict, all_confidences: list[float]):
    bins = metric_data["bins"]
    accuracies = metric_data["accuracies"]
    confidences = metric_data["confidences"]
    counts = metric_data["counts"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ax1.bar(
    #     bins,
    #     [a if a is not None else float("nan") for a in accuracies],
    #     width=0.1,
    #     align="edge",
    #     color="lightblue",
    #     edgecolor="black",
    #     label="accuracy in bin",
    # )
    ax1.plot([0, 1], [0, 1], color="red", linestyle="--", label="perfect calibration")

    x = [c for c in confidences if c is not None]
    y = [a for a, c in zip(accuracies, confidences) if c is not None]
    n = [count for count, c in zip(counts, confidences) if c is not None]

    ax1.plot(x, y, "o-", color="darkblue", label="observed")

    for xi, yi, ni in zip(x, y, n):
        ax1.text(xi, yi + 0.02, f"n={ni}", ha="center", fontsize=8)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.12)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(
        f"Reliability Diagram\n"
        f"ECE = {metric_data['ece']:.4f} ({metric_data["direction"]})"
    )
    ax1.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, 1.02))
    ax1.grid(alpha=0.3)

    ax2.hist(
        all_confidences, bins=20, range=(0, 1), color="lightblue", edgecolor="black"
    )
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Number of fields")
    ax2.set_title(f"Confidence Distribution (n= {len(all_confidences)} fields)")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("assets/reliability_diagram.png", dpi=140)


def calculate_ece_and_draw_chart(field_results: list[tuple]):
    bins = [round(x * 0.1, 1) for x in range(10)]

    field_results_by_bins = [[] for _ in bins]
    metric_data = {
        "bins": bins,
        "accuracies": [],
        "confidences": [],
        "signed": 0,
        "ece": 0,
    }

    for fr in field_results:
        confidence = fr[4]
        index = min(math.floor(confidence * 10), len(bins) - 1)

        field_results_by_bins[index].append(fr)

    for i in range(10):
        if len(field_results_by_bins[i]) == 0:
            metric_data["accuracies"].append(None)
            metric_data["confidences"].append(None)
            continue

        average_confidence = statistics.mean([x[4] for x in field_results_by_bins[i]])
        accuracy = statistics.mean(
            [int(loose(x[2]) == loose(x[3])) for x in field_results_by_bins[i]]
        )
        signed_weight_gap = (len(field_results_by_bins[i]) / len(field_results)) * (
            average_confidence - accuracy
        )

        metric_data["ece"] += abs(signed_weight_gap)
        metric_data["signed"] += signed_weight_gap

        metric_data["accuracies"].append(accuracy)
        metric_data["confidences"].append(average_confidence)

    metric_data["counts"] = [len(fr) for fr in field_results_by_bins]

    sign = (metric_data["signed"] > 0) - (metric_data["signed"] < 0)
    metric_data["direction"] = (
        "+ve" if sign == 1 else ("-ve" if sign == -1 else "neutral")
    )

    all_confidences = [fr[4] for fr in field_results]
    draw_reliablity_diagram(metric_data, all_confidences)

    metric_data["n"] = len(field_results)
    metric_data["accuracy"] = statistics.mean(
        [int(loose(x[2]) == loose(x[3])) for x in field_results]
    )

    return metric_data


def ece_pipeline(subset_len):
    ds = load_cord_dataset()
    sub_dataset = ds.select(range(subset_len))
    result_dict = run_ocr(sub_dataset)
    field_results = []

    for k, v in result_dict.items():
        field_results.extend(emit_field_results_from_receipt(k, v[0], v[1]))

    metric_data = calculate_ece_and_draw_chart(field_results)

    with open("assets/metrics_ece.json", "w") as f:
        json.dump(metric_data, f)
