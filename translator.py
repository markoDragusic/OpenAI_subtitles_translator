from openai import AsyncOpenAI
from charset_normalizer import detect
import tiktoken
import asyncio

client = AsyncOpenAI()
tokens_total = 0

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(100000)
    encoding_method = detect(rawdata)['encoding']

    return encoding_method

def chunk_subtitles(input_file, max_chunk_size=1000):
    # first detect encoding
    encoding_method = detect_encoding(input_file)
    # now read and chunk it
    with open(input_file, 'r', encoding=encoding_method) as file:
        subtitles = file.read().split('\n\n')

    chunks = []
    current_chunk = ''

    for subtitle in subtitles:
        if len(current_chunk) + len(subtitle) + 2 <= max_chunk_size:
            current_chunk += subtitle + '\n\n'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = subtitle + '\n\n'

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
    print('OpenAI finish reason: ', completion.choices[0].finish_reason)
    print('OpenAI usage data: ', completion.usage)

    return completion.choices[0].message.content


async def main():
    input_subtitles = 'subtitle_source.srt'
    # miscalculated it to 8000 initially, then recalculated to 4000 which was almost OK, but an iteration slipped
    # through 4096 margin, thus 3500 should be safe
    chunks = chunk_subtitles(input_subtitles, 3500)

    print('Enter translation language:')
    lang = input()
    print('Translating into...', lang)

    translated_chunks = await asyncio.gather(
        *[translate_chunk(chunk, lang) for chunk in chunks]
    )

    print('Total tokens:', tokens_total )

    translated_srt = '\n\n'.join(translated_chunks)

    with open('subtitle_translated.srt', 'w', encoding='utf-8') as output_file:
        output_file.write(translated_srt)


asyncio.run(main())


