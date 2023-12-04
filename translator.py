from openai import AsyncOpenAI
from charset_normalizer import detect
from time import perf_counter
import tiktoken
import asyncio

import traceback

client = AsyncOpenAI()
tokens_total = 0

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(100000)
    encoding_method = detect(rawdata)['encoding']

    return encoding_method

def chunk_subtitles(input_file, timeframe_chunks, max_chunk_size=1000):
    # first detect encoding
    encoding_method = detect_encoding(input_file)
    # now read and chunk it
    with open(input_file, 'r', encoding=encoding_method) as file:
        subtitles = file.read().split('\n\n')

    chunks = []
    current_chunk = ''

    for subtitle in subtitles:
        split = subtitle.split('\n', 2)

        if len(split) == 3:
            no, time, text = split
        else:
            break
        timeframe_chunks.append({"no": no, "time": time})
        no_timeframe_subtitle_string = f'{no}\n{text}\n\n'

        if len(current_chunk) + len(no_timeframe_subtitle_string) + 2 <= max_chunk_size:
            current_chunk += f'{no}\n{text}\n\n'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = f'{no}\n{text}\n\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

async def translate_chunk(chunk, language):
    global tokens_total

    # OPENAI_API_KEY set manually as environment variable (in my case, I worked with conda virtual env, a bit more complicated to set)
    # Environment OPENAI_API_KEY is automatically used by OpenAI client. You can also provide it as an argument to following 'create' function
    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a subtitles translator. While translating into user desired language keep the same format as input"},
            {"role": "user", "content": "Translate following into " + language + "\n\n" + chunk}
        ]
    )

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(
        "You are a subtitles translator. While translating into user desired language keep the same format as input" \
        + "Translate following into " + language + "\n\n" + chunk + str(completion.choices[0].message)))
    tokens_total += num_tokens

    return completion.choices[0].message.content

def format_translation(translated_content, timeframe_chunks):
    formatted_translation = ''

    translated_parts = translated_content.split('\n\n')
    start_time = perf_counter()

    try:
        for part in translated_parts:
            no, content = part.split('\n', 1)
            matching_time_dict = next((td for td in timeframe_chunks if td['no'] == no), None)

            if matching_time_dict:
                time = matching_time_dict['time']
                formatted_translation += f'{no}\n{time}\n{content}\n\n'
    except ValueError as e:
        print('ValueError: ', e)
        print('Traceback: ', traceback.format_exc())
    except Exception as e:
        print('Exception: ', e)
        print('Traceback: ', traceback.format_exc())

    end_time = perf_counter()
    print('format time:', end_time - start_time)

    return formatted_translation



async def main():
    input_subtitles = 'subtitle_source.srt'
    timeframe_chunks = []
    # miscalculated it to 8000 initially, then recalculated to 4000 which was almost OK, but an iteration slipped
    # through 4096 margin, thus 3500 should be safe
    chunks = chunk_subtitles(input_subtitles, timeframe_chunks, 3500)

    with open('raw_chunks.srt', 'w', encoding='utf-8') as output_file:
        output_file.write(str('\n\n'.join(chunks)))

    print('Enter translation language:')
    lang = input()
    print('Translating into...', lang)

    translated_content = await asyncio.gather(
        *[translate_chunk(chunk, lang) for chunk in chunks]
    )

    translated_content_str = '\n\n'.join(translated_content)
    formatted_translation = format_translation(translated_content_str, timeframe_chunks)

    print('Total tokens:', tokens_total)

    with open('subtitle_translated.srt', 'w', encoding='utf-8') as output_file:
        output_file.write(formatted_translation)

asyncio.run(main())


