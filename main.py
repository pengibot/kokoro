from pathlib import Path

import pandas as pd
import re


def read_chat(file_path: Path) -> pd.DataFrame:
    number_pattern = r'^\d+$'
    timestamp_pattern = r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$'
    blank_lines_pattern = r'^\s*$'
    message_pattern_1 = r'PRESENTED IN JAPANESE BY'
    message_pattern_2 = r'\"THEY\'RE BETTER THAN NOTHING!\"'
    author_pattern = r'ඞSUS SUBSඞ'
    symbol_pattern = r'📱'
    symbol_pattern_sound = r'🔊'
    symbol_pattern_audio = r'♬'
    symbol_pattern_tv = r'📺'
    symbol_pattern_ZWNBSP = r'﻿1'


    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Apply filters to remove unwanted lines
    filtered_lines =[]
    for line in lines:
        line = re.sub('\uFEFF', '', line)
        line = re.sub(r'\{\\an\d}', '', line)
        if(
            not re.search(number_pattern, line) and
            not re.search(timestamp_pattern, line) and
            not re.search(blank_lines_pattern, line) and
            message_pattern_1 not in line and
            message_pattern_2 not in line and
            author_pattern not in line and
            symbol_pattern not in line and
            symbol_pattern_sound not in line and
            symbol_pattern_audio not in line and
            symbol_pattern_tv not in line
        ):
            filtered_lines.append(line.strip())
            # print(line, end='')

    df = pd.DataFrame(filtered_lines)
    return df

all_chats = {}
data_directory = Path("data")
for file in data_directory.glob('*.srt'):
    file_name = file.stem
    all_chats[file_name] = read_chat(file)

# print(all_chats)

text_sequence = ""
for file_name in all_chats.keys():
    text_sequence += " ".join(all_chats[file_name][0])

print(len(text_sequence))