# Gemini MCP-Hero Proxy

Прокси-сервер на FastAPI для интеграции Model Context Protocol (MCP) с Google Gemini API.

Этот сервер выступает в роли моста между клиентским приложением и серверами Google. Он перехватывает официальный формат запросов Google, получает инструменты с удаленных MCP-серверов, конвертирует их в Gemini, и самостоятельно оркестрирует цикл выполнения `function_calling`, стримя финальный результат клиенту.

## Установка и запуск

### Способ 1. Локально (Python VENV)
1. Убедитесь, что установлены зависимости:
```bash
pip install -r requirements.txt
```

2. Запуск сервера:
```bash
python main.py
```
Сервер запустится на `http://0.0.0.0:8000`.

### Способ 2. Через Docker (Рекомендуется для продакшена)
Прокси полностью готов к деплою в Docker-контейнере.

1. Сборка образа:
```bash
docker build -t gemini-mcp-relay .
```

2. Запуск контейнера:
```bash
docker run -d -p 8000:8000 --name mcp-relay gemini-mcp-relay
```

## Настройка (Переменные окружения)

При запуске (как локально, так и в Docker) вы можете переопределить базовый URL, на который прокси будет отправлять запросы.

Для этого задайте переменную окружения `GEMINI_BASE_URL`:

**Пример для Docker:**
```bash
docker run -d -p 8000:8000 -e GEMINI_BASE_URL="https://generativelanguage.googleapis.com" gemini-mcp-proxy
```
*(Если оставить её пустой, прокси автоматически будет стучаться в стандартные API-серверы Google).*

## Использование с клиентского приложения

Сервер полностью эмулирует Google API. Все, что вам нужно сделать на клиенте — это подменить `base_url` на адрес прокси и передать Base64-закодированный заголовок `X-MCP-Servers`.

Пример на Python (официальный `google.genai` SDK):

```python
import base64
import json
from google import genai

# 1. Формируем конфигурацию MCP серверов
# Вы можете подключить несколько серверов. 
# Для приватных серверов можно передать опциональный словарь 'headers' с токенами авторизации.
mcp_config = {
    "math_server": {
        "url": "https://math-mcp.fastmcp.app/mcp"
    },
    "private_database": {
        "url": "https://api.mycompany.com/mcp",
        "headers": {
            "Authorization": "Bearer YOUR_SECRET_TOKEN"
        }
    }
}
mcp_header = base64.b64encode(json.dumps(mcp_config).encode("utf-8")).decode("utf-8")

# 2. Подключаемся к нашему прокси
client = genai.Client(
    api_key="ВАШ_GEMINI_API_KEY", # Ключ уйдет на прокси, а прокси отдаст его в Google
    http_options={
        "base_url": "http://127.0.0.1:8000",
        "headers": {"x-mcp-servers": mcp_header}
    }
)

# 3. Делаем обычный запрос (Прокси сам сходит на MCP, заберет тулы, выполнит их в цикле и вернет ответ)
response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Посчитай квадратный корень из 144, а затем умножь результат на 10. Распиши по шагам.",
)

for chunk in response:
    print(chunk.text, end="", flush=True)
```
