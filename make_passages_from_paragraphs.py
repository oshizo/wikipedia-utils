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
from typing import Callable, Optional

from tqdm import tqdm

from sentence_splitters import MeCabSentenceSplitter

import re
def split_section(text, splitter, max_nchar=750):
    # textを、max_ncharにおおむね収まる、互いに長さが近い複数のチャンクに分割する

    total_nchar = len(text)
    if total_nchar <= max_nchar:
        return [text]

    # チャンク数を見積もる
    nchunk = max(1, round(total_nchar / max_nchar) )
    nchar_chunk = total_nchar/nchunk

    # チャンク当たりの文字数がmax_ncharを超えないようにチャンクを増やす
    while True:
        if nchar_chunk > max_nchar:
            nchunk += 1
            nchar_chunk = total_nchar/nchunk
        else:
            break

    nchar = 0  # section内の文字数を数える
    chunk_id = 1
    sentences = []

    # 文の分割
    for line in re.split(r'\n+', text):
        sentences += splitter(line)
        if len(sentences) > 0:
            sentences[-1] = sentences[-1] + "\n"

    chunks = []
    chunk_text = ""
    for sentence in sentences:
        l = len(sentence)
        
        # チャンク内の文字数が、チャンク番号xチャンク文字数をある程度以上超えそうな場合は次に行く
        if nchar + l//2 >= nchar_chunk*(chunk_id):
            if chunk_id != nchunk:
                chunk_id += 1
                chunks.append(chunk_text.strip())
                chunk_text = ""
            else:
                # 最終チャンクはmax_charを超えてもそこに詰める
                pass

        nchar += l
        chunk_text += sentence
        
    if chunk_text != "":
        chunks.append(chunk_text.strip())
    
    return chunks

def generate_passages(
    paragraphs_file: str,
    max_passage_length: int,
    sentence_splitter: Optional[Callable] = None
):
    
    with gzip.open(paragraphs_file, "rt") as f:
        data = []
        for line in f.readlines():
            data.append(json.loads(line))

        passage_id = 0
        section_rows = []
        lastrow = data[0]

        for row in tqdm(data):
            title = row["title"]
            section = row["section"]

            # titleまたはsectionが変わった場合、そこまでのデータを出力する
            if title != lastrow["title"] or section != lastrow["section"]:

                section_text = "\n".join([r["text"] for r in section_rows])
                for chunk in split_section(section_text, sentence_splitter, max_passage_length):
                    output_item = json.dumps(
                        {
                            "id": passage_id,
                            "pageid": lastrow["pageid"],
                            "revid": lastrow["revid"],
                            "title": lastrow["title"],
                            "section": lastrow["section"],
                            "text": chunk
                        }, ensure_ascii=False)
                    yield output_item

                    passage_id += 1
                section_rows = []

            section_rows.append(row)
            lastrow = row


def main(args: argparse.Namespace):
    sentence_splitter = MeCabSentenceSplitter(args.mecab_option)

    with gzip.open(args.output_file, "wt") as fo:
        passage_generator = generate_passages(
            paragraphs_file=args.paragraphs_file,
            max_passage_length=args.max_passage_length,
            sentence_splitter=sentence_splitter
        )
        for passage_item in tqdm(passage_generator):
            print(json.dumps(passage_item, ensure_ascii=False), file=fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraphs_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_passage_length", type=int, default=750,
        help="It does not take page title lengths into account even if the "
             "--append_title_to_passage_text option is enabled")
    parser.add_argument("--mecab_option", type=str)
    args = parser.parse_args()
    main(args)
