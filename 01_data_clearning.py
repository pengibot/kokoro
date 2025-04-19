import os
from pathlib import Path

import pandas as pd
import re

import unicodedata


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
    symbol_pattern_audio_single = r'♪'
    symbol_pattern_tv = r'📺'
    tips_message = r'TIPS: ko-fi.com/jpsubs'
    dotted_message = r'…'
    symbol_arrow = r'➡'


    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Apply filters to remove unwanted lines
    filtered_lines =[]
    for line in lines:

        line.strip()

        line = re.sub('\uFEFF', '', line)
        line = re.sub(r'\{\\an\d}', '', line)
        line = re.sub(r'<font color="japanese">', '', line)
        line = re.sub(r'\u2009', '', line)
        line = re.sub(dotted_message, '', line)
        line = re.sub(r'\)', '', line)
        line = re.sub(r'\(', '', line)
        line = re.sub(r'！', '', line)
        line = re.sub(r'（', '', line)
        line = re.sub(r'）', '', line)
        line = re.sub(r'？', '', line)
        line = re.sub(r'）', '', line)
        line = re.sub(r'ARIA', '', line)

        if(
            not re.search(number_pattern, line) and
            not re.search(timestamp_pattern, line) and
            not re.search(blank_lines_pattern, line) and
            message_pattern_1 not in line and
            message_pattern_2 not in line and
            author_pattern not in line and
            tips_message not in line
        ):
            line = re.sub(r'\d+', '', line)
            line = re.sub(r' ', '', line)
            line = re.sub(r'　', '', line)
            line = re.sub(r' ', '', line)
            line = re.sub(r' ', '', line)
            line = re.sub(r'\s', '', line)
            line = re.sub(r'\s+', '', line)
            line = re.sub(symbol_arrow, '', line)
            line = re.sub(symbol_pattern_sound, '', line)
            line = re.sub(symbol_pattern_audio, '', line)
            line = re.sub(symbol_pattern_audio_single, '', line)
            line = re.sub(symbol_pattern_tv, '', line)
            line = re.sub(symbol_pattern, '', line)
            line = re.sub(r'[^\S\r\n]+', '', line)
            line = re.sub(r'[a-zA-Z0-9]', '', line)


            line = re.sub(r'《', '', line)
            line = re.sub(r'》', '', line)
            line = re.sub(r'⚟', '', line)
            line = re.sub(r'', '', line)
            line = re.sub(r'', '', line)

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
    text_sequence += "".join(all_chats[file_name][0])

os.makedirs("output", exist_ok=True)
with open("output/combined_text.txt", "w", encoding="utf-8") as f:
    f.write(text_sequence)

print(len(text_sequence))