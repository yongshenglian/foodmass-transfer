"""Build and query a lightweight visual index for standardized ASA24 images.

The descriptor deliberately has no learned dependencies: it combines a foreground-normalized
thumbnail, color histograms, and luminance gradients. This makes it cheap to test whether the
ASA24 reference-photo collection adds useful candidate recall before adding a large vision model.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from PIL import Image


def image_descriptor(path, size=24, bins=16):
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        bbox = Image.fromarray(alpha).getbbox()
        if bbox:
            rgba = rgba.crop(bbox)
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)

    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    thumbnail = pixels.reshape(-1)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, :, channel], bins=bins, range=(0, 1), density=True)
        histograms.extend(hist)
    gray = pixels.mean(axis=2)
    gradient_x = np.diff(gray, axis=1, append=gray[:, -1:])
    gradient_y = np.diff(gray, axis=0, append=gray[-1:, :])
    descriptor = np.concatenate(
        [thumbnail, np.asarray(histograms, dtype=np.float32) * 0.15, gradient_x.reshape(-1), gradient_y.reshape(-1)]
    ).astype(np.float32)
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm else descriptor


def load_image_labels(metadata_xlsx):
    workbook = load_workbook(metadata_xlsx, read_only=True, data_only=True)
    sheet = workbook["AssociatedCodes"]
    header = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    position = {name: index for index, name in enumerate(header)}
    labels = defaultdict(set)
    for row in sheet.iter_rows(min_row=2, values_only=True):
        filename = str(row[position["Image"]])
        code = str(row[position["FoodCode"]]).split(".", 1)[0].zfill(8)
        description = str(row[position["FC_Description"]]).strip()
        labels[filename].add((code, description))
    return labels


def build_index(image_dir, metadata_xlsx, output):
    labels = load_image_labels(metadata_xlsx)
    filenames, descriptors, codes, descriptions = [], [], [], []
    for number, filename in enumerate(sorted(labels), start=1):
        path = os.path.join(image_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            descriptor = image_descriptor(path)
        except Exception as error:
            print(f"Skipping {filename}: {error}", flush=True)
            continue
        pairs = sorted(labels[filename])
        filenames.append(filename)
        descriptors.append(descriptor)
        codes.append("|".join(pair[0] for pair in pairs))
        descriptions.append("|".join(pair[1] for pair in pairs))
        if number % 1000 == 0:
            print(f"Indexed {number}/{len(labels)} metadata images", flush=True)

    np.savez_compressed(
        output,
        descriptors=np.stack(descriptors),
        filenames=np.asarray(filenames),
        codes=np.asarray(codes),
        descriptions=np.asarray(descriptions),
    )
    print(f"Saved {len(filenames)} images to {output}")


class ASA24VisualIndex:
    def __init__(self, index_path):
        data = np.load(index_path)
        self.descriptors = data["descriptors"]
        self.filenames = data["filenames"]
        self.codes = data["codes"]
        self.descriptions = data["descriptions"]

    def search(self, image_path, top_k=20, exclude_filename=None):
        query = image_descriptor(image_path)
        scores = self.descriptors @ query
        order = np.argpartition(scores, -min(top_k + 1, len(scores)))[-min(top_k + 1, len(scores)):]
        order = order[np.argsort(scores[order])[::-1]]
        results = []
        for index in order:
            filename = str(self.filenames[index])
            if exclude_filename and filename == exclude_filename:
                continue
            results.append(
                {
                    "filename": filename,
                    "score": float(scores[index]),
                    "codes": str(self.codes[index]).split("|"),
                    "descriptions": str(self.descriptions[index]).split("|"),
                }
            )
            if len(results) == top_k:
                break
        return results

    def confident_code(self, image_path, exclude_filename=None, margin=0.003):
        """Return a code only when independent visual evidence is locally coherent."""
        neighbors = self.search(image_path, top_k=3, exclude_filename=exclude_filename)
        if not neighbors:
            return None
        top_code = neighbors[0]["codes"][0]
        repeated = sum(top_code in neighbor["codes"] for neighbor in neighbors) >= 2
        separated = len(neighbors) == 1 or neighbors[0]["score"] - neighbors[1]["score"] >= margin
        return top_code if repeated or separated else None

    def ranked_codes(self, image_path, top_k=30, exclude_filename=None):
        """Aggregate nearest reference images into unique code/description candidates."""
        candidates = {}
        for rank, neighbor in enumerate(self.search(image_path, top_k=top_k, exclude_filename=exclude_filename)):
            for code, description in zip(neighbor["codes"], neighbor["descriptions"]):
                score = neighbor["score"] + 1.0 / (rank + 1)
                if code not in candidates or score > candidates[code][0]:
                    candidates[code] = (score, description)
        return [
            {"code": code, "description": description, "score": score}
            for code, (score, description) in sorted(
                candidates.items(), key=lambda item: item[1][0], reverse=True
            )
        ]


def evaluate(index_path, sample_csv, image_dir, top_k):
    index = ASA24VisualIndex(index_path)
    sample = pd.read_csv(sample_csv)
    hit_counts = {1: 0, 5: 0, 10: 0, top_k: 0}
    for _, row in sample.iterrows():
        filename = str(row["Image"])
        true_code = f"{int(row['FoodCode']):08d}"
        neighbors = index.search(os.path.join(image_dir, filename), top_k=top_k, exclude_filename=filename)
        ranked_codes = []
        for neighbor in neighbors:
            for code in neighbor["codes"]:
                if code not in ranked_codes:
                    ranked_codes.append(code)
        rank = ranked_codes.index(true_code) + 1 if true_code in ranked_codes else None
        for cutoff in hit_counts:
            hit_counts[cutoff] += rank is not None and rank <= cutoff
        preview = ", ".join(
            f"{neighbor['codes'][0]}:{neighbor['descriptions'][0][:28]}" for neighbor in neighbors[:5]
        )
        print(f"{filename}: true={true_code} rank={rank} | {preview}")
    for cutoff in sorted(hit_counts):
        print(f"Recall@{cutoff}: {hit_counts[cutoff] / len(sample):.1%}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--image_dir", required=True)
    build.add_argument("--metadata_xlsx", required=True)
    build.add_argument("--output", required=True)
    test = subparsers.add_parser("evaluate")
    test.add_argument("--index", required=True)
    test.add_argument("--sample_csv", required=True)
    test.add_argument("--image_dir", required=True)
    test.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    if args.command == "build":
        build_index(args.image_dir, args.metadata_xlsx, args.output)
    else:
        evaluate(args.index, args.sample_csv, args.image_dir, args.top_k)


if __name__ == "__main__":
    main()
