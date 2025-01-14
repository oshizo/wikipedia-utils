# Copyright 2022 Masatoshi Suzuki (@singletongue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import gzip
import json
from unicodedata import normalize

from bs4 import BeautifulSoup
from logzero import logger
from tqdm import tqdm


DEFAULT_SECTIONS_TO_IGNORE = ["脚注", "出典", "参考文献", "関連項目", "外部リンク"]
DEFAULT_TAGS_TO_REMOVE = ["table"]
# DEFAULT_TAGS_TO_EXTRACT = ["p"]
DEFAULT_TAGS_TO_EXTRACT = ["p", "h3", "h4", "h5", "h6", "dt", "dd", "li", "th", "tr"]
DEFAULT_INNER_TAGS_TO_REMOVE = ["sup"]


def normalize_text(text):
    text = normalize("NFKC", text)
    text = " ".join(text.split())
    text = "".join(char for char in text if char.isprintable())
    text = text.strip()
    return text


def extract_paragraphs_from_html(html, tags_to_extract, tags_to_remove, inner_tags_to_remove):
    soup = BeautifulSoup(html, features="lxml")
    h2 = None
    h3 = None
    h4 = None
    dt = None

    section = soup.find(["section"])
    while section:
        if section.h2 is not None:
            h2 = section.h2.text
            h3 = None
            h4 = None
            dt = None

        for tag in section.find_all(tags_to_remove):
            tag.clear()

        tags = section.find_all(tags_to_extract)
        for tag in tags:

            # これらのタグの配下の場合、先に親要素のテキストが取得されているためスキップする
            skip = False
            for parent_tag in ["dd", "li", "th", "tr"]:
                if tag.find_parent(parent_tag):
                    skip = True
                    break
            if skip:
                continue

            for inner_tag in tag.find_all(inner_tags_to_remove):
                inner_tag.clear()

            # h3→h4→dtの順序の階層を前提に、上位の階層が変化したら下位の階層をリセットする
            if tag.name == "h3":
                h3 = tag.text
                h4 = None
                dt = None
                continue

            if tag.name == "h4":
                h4 = tag.text
                dt = None
                continue

            if tag.name == "dt":
                dt = tag.text
                continue

            paragraph_text = normalize_text(tag.text)
            yield ([h2, h3, h4, dt], paragraph_text, tag.name)

        section = section.find_next_sibling(["section"])


def main(args):
    if args.tags_to_extract is not None:
        tags_to_extract = args.tags_to_extract
    else:
        tags_to_extract = DEFAULT_TAGS_TO_EXTRACT

    if args.tags_to_remove is not None:
        tags_to_remove = args.tags_to_remove
    else:
        tags_to_remove = DEFAULT_TAGS_TO_REMOVE

    if args.inner_tags_to_remove is not None:
        inner_tags_to_remove = args.inner_tags_to_remove
    else:
        inner_tags_to_remove = DEFAULT_INNER_TAGS_TO_REMOVE

    if args.sections_to_ignore is not None:
        sections_to_ignore = args.sections_to_ignore
    else:
        sections_to_ignore = DEFAULT_SECTIONS_TO_IGNORE

    logger.info("tags_to_extract: %s", tags_to_extract)
    logger.info("tags_to_remove: %s", tags_to_remove)
    logger.info("inner_tags_to_remove: %s", inner_tags_to_remove)
    logger.info("sections_to_ignore: %s", sections_to_ignore)

    with gzip.open(args.page_htmls_file, "rt") as f, gzip.open(args.output_file, "wt") as fo:
        for line in tqdm(f):
            input_item = json.loads(line.rstrip("\n"))
            page_id = input_item["pageid"]
            rev_id = input_item["revid"]
            title = input_item["title"]
            html = input_item["html"]

            paragraph_index = 0
            for item in extract_paragraphs_from_html(html, tags_to_extract, tags_to_remove, inner_tags_to_remove):
                sections, paragraph_text, tag_name = item

                if sections[0] in sections_to_ignore:
                    continue
                if len(paragraph_text) < args.min_paragraph_length:
                    continue
                if len(paragraph_text) > args.max_paragraph_length:
                    continue

                sections_dict = {}
                if sections[0] is not None:
                    sections_dict["h2"] = sections[0]
                if sections[1] is not None:
                    sections_dict["h3"] = sections[1]
                if sections[2] is not None:
                    sections_dict["h4"] = sections[2]
                if sections[3] is not None:
                    sections_dict["dt"] = sections[3]


                output_item = {
                    "id": "{}-{}-{}".format(page_id, rev_id, paragraph_index),
                    "pageid": page_id,
                    "revid": rev_id,
                    "paragraph_index": paragraph_index,
                    "title": title,
                    "section": sections_dict,
                    "text": paragraph_text,
                    "html_tag": tag_name,
                }
                print(json.dumps(output_item, ensure_ascii=False), file=fo)
                paragraph_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_htmls_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--tags_to_extract", nargs="+", type=str)
    parser.add_argument("--tags_to_remove", nargs="+", type=str)
    parser.add_argument("--inner_tags_to_remove", nargs="+", type=str)
    parser.add_argument("--sections_to_ignore", nargs="+", type=str)
    parser.add_argument("--min_paragraph_length", type=int, default=10)
    parser.add_argument("--max_paragraph_length", type=int, default=1000)
    args = parser.parse_args()
    main(args)
