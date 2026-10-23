"""Reproduce the 2,500 hardest selected synthetic calibration texts offline.

No network, private project files or third-party text is used. Run this file
from any directory; the default destination is the packaged corpus directory.
The original 10,000-record pool and checked-in native count receipt reproduce
the selection. Native counts select this pool once; consumers calibrate the
selected 2,500 texts with their own model and prefix policy.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

CATEGORIES = ("code", "documentation", "prose", "structured", "multilingual")
SOURCE_COUNT = 2000
COUNT = 500
SCHEMA = "runtime-llama-calibration-corpus-v2"
SOURCE_CORPUS_SHA256 = (
    "6a4733cf1d175b8b5b254f5951d19c5e4e5f5db2da8eaa1e15904474c53d3035"
)
SELECTION_RECEIPT = Path(__file__).with_name("calibration-hard-selection.json")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _code(rng, index):
    names = ["parcel", "reading", "segment", "ticket", "sample", "bucket", "route"]
    name = rng.choice(names)
    threshold, scale, offset = (
        rng.randint(2, 97),
        rng.randint(2, 19),
        rng.randint(1, 991),
    )
    samples = [rng.randint(-500, 500) for _ in range(3 + index % 11)]
    templates = [
        (
            "python",
            f"def normalize_{name}(values, threshold={threshold}):\n"
            f"    scale = {scale}\n    offset = {offset}\n"
            "    result = []\n    for position, value in enumerate(values):\n"
            "        if value is None or abs(value) < threshold:\n            continue\n"
            "        result.append((position, (value + offset) / scale))\n"
            f"    return sorted(result, key=lambda item: item[1])\n\n"
            f"example = normalize_{name}({samples!r})\n",
        ),
        (
            "javascript",
            f"export async function collect{name.title()}(source, limit = {threshold}) {{\n"
            "  const result = new Map();\n  for await (const record of source) {\n"
            f"    const score = (record.value + {offset}) / {scale};\n"
            "    if (Number.isFinite(score)) result.set(record.key, score);\n"
            "    if (result.size >= limit) break;\n  }\n"
            "  return [...result].sort((left, right) => left[1] - right[1]);\n}\n",
        ),
        (
            "go",
            f"func Filter{name.title()}(values []int) ([]int, error) {{\n"
            "    result := make([]int, 0, len(values))\n"
            "    for _, value := range values {\n"
            f'        if value < -{offset} {{ return nil, fmt.Errorf("value out of range") }}\n'
            f"        if value % {scale} == 0 && value > {threshold} {{\n"
            "            result = append(result, value)\n        }\n    }\n"
            "    return result, nil\n}\n",
        ),
        (
            "rust",
            f"fn sum_{name}(values: &[i64]) -> Result<i64, &'static str> {{\n"
            f"    values.iter().filter(|value| **value >= {threshold}).try_fold(\n"
            f"        {offset}_i64, |total, value| total.checked_add(value * {scale})\n"
            '            .ok_or("numeric overflow"),\n    )\n}\n',
        ),
        (
            "sql",
            f"WITH selected AS (\n  SELECT category, amount, observed_at\n"
            f"  FROM {name}_measurements\n  WHERE amount BETWEEN {threshold} AND {offset + 1000}\n"
            ")\nSELECT category, COUNT(*) AS samples, AVG(amount) AS mean_amount\n"
            f"FROM selected GROUP BY category HAVING COUNT(*) >= {scale}\n"
            "ORDER BY mean_amount DESC;\n",
        ),
        (
            "cpp",
            f"std::vector<double> adjust_{name}(std::span<const double> input) {{\n"
            "    std::vector<double> output;\n    output.reserve(input.size());\n"
            "    for (const auto value : input) {\n"
            f"        if (std::isfinite(value) && value > {threshold})\n"
            f"            output.push_back((value + {offset}.0) / {scale}.0);\n"
            "    }\n    return output;\n}\n",
        ),
    ]
    language, body = templates[index % len(templates)]
    if index % 5 == 0:
        return language, body
    if index % 5 >= 3:
        for part in range(1 + index % 4):
            values = [rng.randint(-999, 999) for _ in range(8 + part * 3)]
            body += (
                f"\n/* Scenario {part + 1}: preserve order, empty inputs, duplicates, "
                f"and values around boundary {threshold + part}. Test data: {values}. */\n"
            )
    return language, body


def _documentation(rng, index):
    subject = rng.choice(
        ["queue", "archive", "sensor", "catalog", "journal", "scheduler"]
    )
    capacity, delay = rng.randint(8, 4096), rng.randint(1, 180)
    sections = [
        f"# Operating the {subject}\n\nThe configured capacity is {capacity} records. "
        f"A retry waits at most {delay} seconds before returning an explicit failure.",
        "## Preconditions\n\n- Verify the selected profile and available storage.\n"
        "- Keep the input sequence immutable while work is pending.\n"
        "- Stop admitting work before draining existing requests.",
        "## Recovery\n\nIf a worker fails, retain its unacknowledged records. "
        "Retry only records without a committed result; a repeated acknowledgement "
        "must not create a second output entry.",
        f"| Setting | Value | Meaning |\n|---|---:|---|\n"
        f"| capacity | {capacity} | maximum retained records |\n"
        f"| retry_seconds | {delay} | bounded retry interval |\n"
        "| ordering | stable | original source positions |",
        "```text\ncreated -> queued -> running -> completed\n"
        "                      -> failed\n                      -> cancelled\n```\n"
        "A terminal state must remain observable until its owner acknowledges it.",
        "### Acceptance example\n\nSubmit three distinct records and one repeated "
        "record. Expect four results in input order. Cancel a second request while "
        "the first request drains; only the cancelled owner should lose admission.",
        "## Troubleshooting\n\nAn empty result differs from an unavailable service. "
        "Report the observed capability, the rejected operation and its bounded "
        "diagnostic. Preserve Unicode text without normalizing source offsets.",
    ]
    heading = sections.pop(0)
    rng.shuffle(sections)
    count = (1, 2, 3, 5, 7)[index % 5]
    return "en", "\n\n".join([heading, *sections[: count - 1]])


def _prose(rng, index):
    places = ["harbor", "library", "orchard", "workshop", "observatory", "station"]
    people = ["the cartographer", "the gardener", "the mechanic", "the archivist"]
    place, person = rng.choice(places), rng.choice(people)
    number, hour = rng.randint(3, 999), rng.randint(1, 23)
    paragraphs = [
        f"At the {place}, {person} counted {number} small markers before the clock "
        f"reached {hour:02d}:00. The count mattered less than the pattern: every "
        "third marker faced the window, while the others followed the old road.",
        "The weather changed without warning. A bright strip of sky appeared "
        "above the roof, and the sound of the afternoon crowd faded into the "
        "steady tapping of water against the shutters.",
        '"Keep the first version," someone said. "We may need to understand '
        'why the new arrangement works." The reply was a quiet laugh, followed '
        "by a careful sketch on the back of an unused inventory sheet.",
        "No single observation settled the question. The group compared notes, "
        "separated facts from guesses, and marked the places where the evidence "
        "was still thin. They agreed to repeat the measurement after sunset.",
        "One basket held smooth stones; another held folded paper. Their labels "
        "had faded, but the shapes remained distinct. Reconstructing the route "
        "required both the labels and the order in which the objects had arrived.",
        "The final report used ordinary words. It described what had changed, "
        "which checks had passed, and what remained uncertain. Anyone returning "
        "a week later could follow the reasoning without meeting its authors.",
    ]
    head = paragraphs.pop(0)
    rng.shuffle(paragraphs)
    return "en", "\n\n".join([head, *paragraphs[: (0, 1, 2, 3, 5)[index % 5]]])


def _structured(rng, index):
    count, offset = rng.randint(2, 12), rng.randint(10, 9999)
    names = rng.sample(["amber", "birch", "cedar", "delta", "elm", "fern", "grove"], 4)
    rows = [
        {
            "name": names[row % len(names)],
            "sequence": offset + row,
            "value": round(rng.uniform(-15, 150), 3),
            "active": bool(row % 3),
            "labels": ["synthetic", rng.choice(["cold", "warm", "stable", "pending"])],
        }
        for row in range(count + (index % 5) * 4)
    ]
    kind = index % 5
    if kind == 0:
        return "json", _json({"schema": 1, "rows": rows, "next": None})
    if kind == 1:
        return "csv", "name,sequence,value,active\n" + "\n".join(
            f"{row['name']},{row['sequence']},{row['value']},{str(row['active']).lower()}"
            for row in rows
        )
    if kind == 2:
        return "yaml", "records:\n" + "\n".join(
            f"  - name: {row['name']}\n    sequence: {row['sequence']}\n"
            f"    value: {row['value']}\n    active: {str(row['active']).lower()}"
            for row in rows
        )
    if kind == 3:
        return "xml", "<measurements>\n" + "\n".join(
            f'  <sample name="{row["name"]}" sequence="{row["sequence"]}">'
            f"<value>{row['value']}</value><active>{str(row['active']).lower()}</active></sample>"
            for row in rows
        ) + "\n</measurements>"
    return "log", "\n".join(
        f"level={rng.choice(['info', 'debug', 'warning'])} component={row['name']} "
        f"sequence={row['sequence']} value={row['value']} "
        f'message="synthetic observation acknowledged"'
        for row in rows
    )


_LANGUAGES = {
    "it": (
        "La biblioteca conserva mappe, registri e racconti del quartiere.",
        "Ogni frammento mantiene il collegamento alla pagina originale.",
        "Prima della consegna verifichiamo ordine, completezza e caratteri accentati.",
        "La luce del mattino attraversa il giardino e raggiunge la finestra aperta.",
    ),
    "fr": (
        "La bibliothèque conserve des cartes et des récits du quartier.",
        "Chaque fragment garde son lien avec le document d’origine.",
        "Nous vérifions les résultats avant de publier le rapport détaillé.",
        "La lumière du matin traverse le jardin et éclaire la fenêtre ouverte.",
    ),
    "de": (
        "Die Bibliothek bewahrt Karten und Erzählungen aus dem Viertel auf.",
        "Jeder Abschnitt behält seine Verbindung zum ursprünglichen Dokument.",
        "Vor der Übergabe prüfen wir Reihenfolge, Vollständigkeit und Sonderzeichen.",
        "Das Morgenlicht fällt durch den Garten auf das offene Fenster.",
    ),
    "es": (
        "La biblioteca conserva mapas y relatos de la comunidad.",
        "Cada fragmento mantiene su relación con el documento original.",
        "Antes de la entrega comprobamos el orden y los caracteres acentuados.",
        "La luz de la mañana cruza el jardín y alcanza la ventana abierta.",
    ),
    "pt": (
        "A biblioteca conserva mapas e histórias do bairro.",
        "Cada trecho mantém a ligação com o documento original.",
        "Antes da entrega verificamos a ordem e os caracteres acentuados.",
        "A luz da manhã atravessa o jardim e chega à janela aberta.",
    ),
    "ja": (
        "図書館には地域の地図と物語が保存されています。",
        "各断片は元の文書とのつながりを保持します。",
        "結果を渡す前に順序と完全性を確認します。",
        "朝の光が庭を通り、開いた窓に届きます。",
    ),
    "zh": (
        "图书馆保存着社区的地图和故事。",
        "每个片段都保留与原始文档的联系。",
        "提交结果之前，我们检查顺序和完整性。",
        "清晨的阳光穿过花园，照进打开的窗户。",
    ),
    "ar": (
        "تحتفظ المكتبة بخرائط الحي وقصصه.",
        "يحتفظ كل مقطع بصلته بالمستند الأصلي.",
        "نتحقق من الترتيب والاكتمال قبل تسليم النتائج.",
        "يعبر ضوء الصباح الحديقة ويصل إلى النافذة المفتوحة.",
    ),
    "ru": (
        "Библиотека хранит карты и рассказы о районе.",
        "Каждый фрагмент сохраняет связь с исходным документом.",
        "Перед передачей результатов мы проверяем порядок и полноту.",
        "Утренний свет проходит через сад и достигает открытого окна.",
    ),
    "hi": (
        "पुस्तकालय में मोहल्ले के नक्शे और कहानियाँ सुरक्षित हैं।",
        "हर अंश मूल दस्तावेज़ से अपना संबंध बनाए रखता है।",
        "परिणाम देने से पहले हम क्रम और पूर्णता की जाँच करते हैं।",
        "सुबह की रोशनी बगीचे से होकर खुली खिड़की तक पहुँचती है।",
    ),
}


def _multilingual(rng, index):
    language = tuple(_LANGUAGES)[index % len(_LANGUAGES)]
    sentences = list(_LANGUAGES[language])
    rng.shuffle(sentences)
    number = rng.randint(10, 99999)
    marker = rng.choice(["α", "β", "γ", "δ", "🌿", "🧭", "é", "e\u0301"])
    # Numeric fields and notation are ordinary document content, not ID suffixes.
    lines = [f"{marker} {number}: {sentences[0]}"]
    for part in range((1, 2, 4, 7, 11)[index % 5]):
        lines.append(
            f"{part + 1}. {sentences[part % len(sentences)]} "
            f"[{number + part * rng.randint(1, 37)}; {rng.randint(1, 99)}%]"
        )
    if index % 4 == 0:
        second = rng.choice([value for value in _LANGUAGES if value != language])
        lines.append(_LANGUAGES[second][rng.randrange(4)])
    return language, "\n".join(lines)


_GENERATORS = dict(
    zip(CATEGORIES, (_code, _documentation, _prose, _structured, _multilingual))
)


def original_records():
    """Reconstruct the unchanged original pool, without reading source files."""
    result = {}
    texts = set()
    for category_index, category in enumerate(CATEGORIES):
        records = []
        for index in range(SOURCE_COUNT):
            rng = random.Random(73471 + category_index * 100000 + index)
            language, text = _GENERATORS[category](rng, index)
            if not text or text in texts:
                raise RuntimeError(f"duplicate/empty corpus text: {category}/{index}")
            texts.add(text)
            records.append(
                {
                    "id": f"{category}-{index + 1:04d}",
                    "category": category,
                    "language": language,
                    "text": text,
                }
            )
        result[category] = records
    return result


def _file_description(category, records):
    payload = ("\n".join(_json(record) for record in records) + "\n").encode("utf-8")
    return payload, {
        "file": category + ".jsonl",
        "count": len(records),
        "total_characters": sum(len(record["text"]) for record in records),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def select_hardest(originals, receipt):
    """Verify source/count provenance and select with exact rational ordering."""
    descriptions = {
        category: _file_description(category, originals[category])[1]
        for category in CATEGORIES
    }
    source_digest = hashlib.sha256(_json(descriptions).encode("utf-8")).hexdigest()
    if (
        receipt.get("schema") != "runtime-llama-hard-corpus-selection-v1"
        or source_digest != SOURCE_CORPUS_SHA256
        or receipt.get("source_corpus_sha256") != source_digest
        or receipt.get("selected_per_category") != COUNT
    ):
        raise ValueError("selection receipt does not match the original corpus")
    by_id = {
        record["id"]: record
        for category in CATEGORIES
        for record in originals[category]
    }
    measurements = receipt.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(by_id):
        raise ValueError("selection requires every original native token count")
    seen = set()
    for row in measurements:
        original = by_id.get(row.get("id"))
        if (
            original is None
            or row["id"] in seen
            or row.get("category") != original["category"]
            or row.get("source_characters") != len(original["text"])
            or row.get("text_sha256")
            != hashlib.sha256(original["text"].encode("utf-8")).hexdigest()
            or type(row.get("tokens")) is not int
            or row["tokens"] <= 0
        ):
            raise ValueError("invalid original native measurement")
        seen.add(row["id"])
    result = {}
    for category in CATEGORIES:
        ranked = sorted(
            (row for row in measurements if row["category"] == category),
            key=lambda row: (
                Fraction(row["source_characters"], row["tokens"]),
                row["id"],
            ),
        )
        chosen = [row["id"] for row in ranked[:COUNT]]
        if receipt.get("selected", {}).get(category) != chosen:
            raise ValueError("selection is not the lowest measured ratio per category")
        result[category] = [
            {**by_id[identifier], "id": f"{category}-{index + 1:04d}"}
            for index, identifier in enumerate(chosen)
        ]
    return result


def generate(destination: Path, *, selection_receipt: Path = SELECTION_RECEIPT):
    originals = original_records()
    selection_payload = selection_receipt.read_bytes()
    receipt = json.loads(selection_payload)
    selected = select_hardest(originals, receipt)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for category in CATEGORIES:
        records = selected[category]
        payload, description = _file_description(category, records)
        filename = category + ".jsonl"
        (destination / filename).write_bytes(payload)
        files[category] = description
    corpus_sha256 = hashlib.sha256(_json(files).encode("utf-8")).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "sample_count": COUNT * len(CATEGORIES),
        "corpus_sha256": corpus_sha256,
        "categories": files,
        "provenance": "Original synthetic text generated without external sources.",
        "character_unit": "Unicode code points as counted by Python len(str).",
        "selection": {
            "source_corpus_sha256": SOURCE_CORPUS_SHA256,
            "source_sample_count": SOURCE_COUNT * len(CATEGORIES),
            "selected_per_category": COUNT,
            "receipt_sha256": hashlib.sha256(selection_payload).hexdigest(),
            "metric": "lowest source Unicode characters / fully prepared native tokens",
            "tie_break": "original record ID ascending",
            "measurement_binding": receipt["binding"],
        },
    }
    (destination / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/runtime_llama/calibration_corpus",
    )
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
