import telebot
import uuid
import whisper
import os
import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from settings import CONTENT_TYPES, API_KEY

def predicts_handler(predicts):
    emotions = ["нейтральность", "радость", "печаль", "страх", "злость", "стыд", "веселье"]
    preds = []
    for i in predicts:
        preds.append(float(i))
    max_index = preds.index(max(preds))
    print(preds)
    return emotions[max_index]

def recognize_text(filename):
    model = whisper.load_model("small")
    result = model.transcribe(filename)
    return result["text"]

def recognize_emote(text):

    # Загрузка обученной модели
    model_ = BertForSequenceClassification.from_pretrained("./emote_analyse")
    tokenizer_ = AutoTokenizer.from_pretrained("./emote_analyse")
    sentence = text
    tokens = []
    mask = []
    # Токенизация
    encoded_sentence = tokenizer_.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    tokens.append(encoded_sentence['input_ids'])
    mask.append(encoded_sentence['attention_mask'])

    tokens = torch.cat(tokens, dim=0)
    mask = torch.cat(mask, dim=0)

    outputs = model_(tokens, token_type_ids=None,
                     attention_mask=mask)
    predicts = outputs[0][0]
    return predicts

bot = telebot.TeleBot(API_KEY)
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, 'Привет! Я бот, который может распознать речь и эмоции голосового сообщения. Скажи мне что-нибудь!')
@bot.message_handler(content_types=CONTENT_TYPES)
def handle_start(message):
    bot.send_message(message.chat.id, 'Извини, я распознаю только голосовые сообщения. Попробуй записать мне что-нибудь!')
@bot.message_handler(content_types=['voice'])
def voice_processing(message):

    filename = str(uuid.uuid4())
    file_name_full="./voice/"+filename+".ogg"
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(file_name_full, 'wb') as new_file:
        new_file.write(downloaded_file)
    text="«"+recognize_text(file_name_full)+"»"
    bot.reply_to(message, text)
    os.remove(file_name_full)
    predicts= recognize_emote(text)
    emote = predicts_handler(predicts)
    bot.reply_to(message, emote)
bot.polling()