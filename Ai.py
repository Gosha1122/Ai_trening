import os
import shutil
import glob
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ================== КОНФИГУРАЦИЯ ==================
MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"  # базовая модель
OUTPUT_DIR = "./parody_dialog_model"                  # папка для сохранения
TRAIN_FILE = "messages_output.txt"                          # файл с обучающими диалогами
BLOCK_SIZE = 128                                      # длина блока токенов
EPOCHS = 5                                            # количество эпох
BATCH_SIZE = 4                                        # размер батча
LEARNING_RATE = 5e-5
WARMUP_STEPS = 800
LOGGING_STEPS = 100
SAVE_STEPS = 200                                      # сохранять чекпоинт каждые N шагов
MAX_LENGTH_RESPONSE = 100                             # максимальная длина ответа
TEMPERATURE = 0.8                                     # температура генерации
TOP_K = 40
TOP_P = 0.9
REPETITION_PENALTY = 1.2
HISTORY_DEPTH = 5                                    # сколько последних реплик учитывать

# ================== ПОДГОТОВКА ДАННЫХ ==================
def prepare_dialog_dataset(file_path, tokenizer):
    """
    Читает файл с диалогами (каждая реплика с новой строки, диалоги разделены пустой строкой).
    Возвращает список токенизированных диалогов.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dialogues = []
    current = []
    for line in lines:
        line = line.strip()
        if line == "":
            if current:
                dialogues.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        dialogues.append("\n".join(current))

    tokenized_dialogues = [tokenizer.encode(d) for d in dialogues]
    return tokenized_dialogues

class DialogDataset(torch.utils.data.Dataset):
    """Кастомный Dataset: разбивает диалоги на блоки фиксированной длины."""
    def __init__(self, tokenized_dialogues, block_size):
        self.examples = []
        for tokens in tokenized_dialogues:
            # Разбиваем на блоки по block_size с перекрытием через шаг block_size
            for i in range(0, len(tokens) - block_size + 1, block_size):
                self.examples.append(tokens[i:i + block_size])
            # Добавляем остаток (если есть)
            if len(tokens) % block_size != 0:
                self.examples.append(tokens[len(tokens) - (len(tokens) % block_size):])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

# ================== ОБУЧЕНИЕ ==================
def train_model():
    """
    Обучает (или дообучает) модель. Если в OUTPUT_DIR есть чекпоинты, загружает веса из последнего
    и продолжает обучение. После завершения сохраняет финальную модель в корень OUTPUT_DIR.
    """
    # 1. Загружаем токенизатор и модель из базовой модели
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # 2. Добавляем специальные токены
    special_tokens = {"pad_token": "<pad>", "bos_token": "<s>", "eos_token": "</s>"}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Проверяем наличие чекпоинтов в OUTPUT_DIR
    checkpoints = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"Найден чекпоинт: {latest_checkpoint}. Продолжаем обучение с сохранённых весов.")
        # Загружаем модель из чекпоинта (веса)
        model = GPT2LMHeadModel.from_pretrained(latest_checkpoint)
        # Токенизатор создаём заново из базовой модели и добавляем спецтокены
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        print("Модель загружена из чекпоинта.")
    else:
        print("Чекпоинтов нет, начинаем обучение с нуля.")
        # Если папка OUTPUT_DIR существует, но без чекпоинтов (например, от неудачного запуска), очистим её
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    # 4. Подготовка данных
    tokenized_dialogues = prepare_dialog_dataset(TRAIN_FILE, tokenizer)
    dataset = DialogDataset(tokenized_dialogues, BLOCK_SIZE)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Параметры обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,              # хранить не более 2 последних чекпоинтов
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        fp16=torch.cuda.is_available(),  # использовать mixed precision при наличии GPU
        dataloader_pin_memory=False,
        prediction_loss_only=True,
    )

    # 6. Создание Trainer и запуск обучения
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        # параметр resume_from_checkpoint не используется, так как в новых версиях transformers он может отсутствовать
    )

    trainer.train()

    # 7. Сохранение финальной модели и токенизатора
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Финальная модель сохранена в {OUTPUT_DIR}")

# ================== ГЕНЕРАЦИЯ ОТВЕТА ==================
def generate_response(model, tokenizer, user_input, history):
    """
    Генерирует ответ на основе пользовательского ввода и истории диалога.
    history – список кортежей (user_msg, bot_msg).
    """
    # Формируем контекст из последних HISTORY_DEPTH обменов
    context = ""
    for user_msg, bot_msg in history[-HISTORY_DEPTH:]:
        context += f"Пользователь: {user_msg}\nЧеловек: {bot_msg}\n"
    context += f"Пользователь: {user_input}\nЧеловек:"

    input_ids = tokenizer.encode(context, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + MAX_LENGTH_RESPONSE,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
        )

    # Декодируем только сгенерированную часть
    generated = output[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    # Отрезаем всё после первого переноса строки (чтобы не залезть в следующую реплику)
    response = response.split("\n")[0].strip()
    return response

# ================== ИНТЕРАКТИВНЫЙ ЧАТ ==================
def chat():
    """Загружает обученную модель и запускает чат в консоли."""
    tokenizer = GPT2Tokenizer.from_pretrained(OUTPUT_DIR)
    model = GPT2LMHeadModel.from_pretrained(OUTPUT_DIR)
    model.eval()

    print("Диалоговый ИИ готов! Введите 'выход' для завершения.\n")
    history = []

    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            break
        if not user_input.strip():
            continue

        bot_response = generate_response(model, tokenizer, user_input, history)
        print(f"Бот: {bot_response}\n")
        history.append((user_input, bot_response))

# ================== ПРОВЕРКА НАЛИЧИЯ МОДЕЛИ ==================
def model_exists(directory):
    """Проверяет, есть ли в папке финальная модель (pytorch_model.bin или model.safetensors)."""
    return os.path.exists(os.path.join(directory, "pytorch_model.bin")) or \
           os.path.exists(os.path.join(directory, "model.safetensors"))

# ================== ЗАПУСК ==================
if __name__ == "__main__":
    if model_exists(OUTPUT_DIR):
        print("Найдена обученная модель, перехожу к чату.")
        chat()
    else:
        print("Модель не найдена. Запускаю обучение (или продолжение с чекпоинта).")
        train_model()
        print("Обучение завершено. Запускаю чат.")
        chat()